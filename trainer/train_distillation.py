"""
MiniMind 知识蒸馏训练脚本

知识蒸馏是一种将大模型（教师模型）的知识迁移到小模型（学生模型）的技术。
核心思想：
- 教师模型通常更大更强，输出包含"软标签"信息
- 学生模型学习教师模型的输出分布，而非硬标签
- 软标签包含类别间的相似性信息，比硬标签更丰富

蒸馏损失：
- CE Loss：传统交叉熵，使用真实标签
- KL Divergence Loss：蒸馏损失，让学生模型学习教师模型的输出分布

总损失 = alpha * CE_loss + (1-alpha) * KL_loss

温度参数T：
- T越大，概率分布越平滑
- T越小，概率分布越尖锐
- 推理时用T=1恢复正常分布
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏损失（KL散度）

    学生学习教师的软概率分布：
    1. 教师 logits 经过 softmax/T 得到软标签（不梯度）
    2. 学生 logits 经过 log_softmax/T 和 softmax/T 计算 KL 散度
    3. 乘以 T^2 补偿温度带来的梯度缩放

    参数：
        student_logits: 学生模型的logits
        teacher_logits: 教师模型的logits
        temperature: 温度参数，越大分布越平滑
        reduction: 损失聚合方式

    返回：
        蒸馏损失
    """
    with torch.no_grad():
        # 教师概率分布（软标签）
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生对数概率
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL散度：sum(p_teacher * log(p_teacher/p_student))
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )

    # 乘以 T^2 补偿（来自链式法则）
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """
    知识蒸馏训练单个epoch

    参数：
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 每个epoch的迭代数
        teacher_model: 教师模型（冻结，不更新）
        lm_config_student: 学生模型配置
        start_step: 起始步数
        wandb: 实验追踪工具
        alpha: CE损失权重，总损失 = alpha*CE + (1-alpha)*KL
        temperature: 蒸馏温度
    """
    start_time = time.time()

    # 教师模型设为eval模式并冻结
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # -------------------- 数据准备 --------------------
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 创建loss掩码：只计算非padding位置的loss
        loss_mask = (labels[..., 1:] != -100).float()

        # -------------------- 学习率调度 --------------------
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------------------- 学生模型前向传播 --------------------
        with autocast_ctx:
            res = model(input_ids)
            # 获取学生logits（预测下一个token）
            student_logits = res.logits[..., :-1, :].contiguous()

        # -------------------- 教师模型前向传播（不梯度） --------------------
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 确保学生和教师的vocab维度一致
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # -------------------- 计算损失 --------------------
        # 1) Ground-Truth 交叉熵损失
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)

        # 只在有效位置计算loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)

        # MoE模型需要加上辅助损失
        if lm_config_student.use_moe:
            ce_loss = ce_loss_raw + res.aux_loss
        else:
            ce_loss = ce_loss_raw

        # 2) 蒸馏损失：学生学习教师的输出分布
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * KL
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # -------------------- 反向传播 --------------------
        scaler.scale(loss).backward()

        # -------------------- 梯度更新 --------------------
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # -------------------- 日志输出 --------------------
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                  f'loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, '
                  f'aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, '
                  f'learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # -------------------- 模型保存 --------------------
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer,
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 知识蒸馏")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default=None, help="训练设备，默认为自动检测")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # 自动检测设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = f"cuda:{local_rank}" if dist.is_initialized() else "cuda:0"
        elif torch.backends.mps.is_available():
            args.device = "mps:0"
        else:
            args.device = "cpu"
    elif dist.is_initialized() and "cuda" in args.device:
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.use_moe)
    )
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else ("mps" if "mps" in args.device else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "mps" and args.dtype == "float16":
        Logger("MPS 设备不支持 float16 混合精度，自动切换为 bfloat16")
        dtype = torch.bfloat16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配置Wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义学生和教师模型 ==========
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 教师模型：通常比学生模型大
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()                              # 教师模型设为eval模式
    teacher_model.requires_grad_(False)               # 冻结，不计算梯度
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda" and dtype == torch.float16))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        if device_type == "cuda" and 'scaler' in ckp_data:
            scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student,
                       start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student,
                       0, wandb, args.alpha, args.temperature)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
