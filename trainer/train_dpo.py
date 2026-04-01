"""
MiniMind DPO（Direct Preference Optimization）训练脚本

DPO是一种直接优化偏好的强化学习算法，用于对齐语言模型与人类偏好。
相比传统的RLHF（Reinforcement Learning with Human Feedback），DPO：
- 不需要训练单独的奖励模型
- 不需要复杂的PPO采样过程
- 训练更稳定，样本效率更高

DPO核心思想：
- 收集同一prompt的两个回复（chosen/rejected）
- chosen是偏好的，rejected是不偏好的
- 优化目标：让模型倾向于生成chosen，规避rejected

DPO损失函数：
    loss = -log(sigmoid(beta * (log_prob_chosen - log_prob_rejected
                              - (ref_log_prob_chosen - ref_log_prob_rejected))))

其中：
- log_prob_chosen: 策略模型对chosen回复的log概率
- log_prob_rejected: 策略模型对rejected回复的log概率
- ref_log_prob_*: 参考模型的log概率（通常是SFT模型）
- beta: 温度参数，控制偏离参考模型的程度
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
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将logits转换为每个token的log概率

    参数：
        logits: 模型输出的logits，形状 (batch_size, seq_len, vocab_size)
        labels: token ID序列，形状 (batch_size, seq_len)

    返回：
        log_probs: 每个位置的log概率，形状 (batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=2)
    # gather：选择每个位置对应token的log概率
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算DPO损失

    公式：
        π_logratios = log(π(y_c)/π(y_r)) - log(π_ref(y_c)/π_ref(y_r))
        loss = -log(sigmoid(beta * π_logratios))

    参数：
        ref_log_probs: 参考模型的log概率（chosen和rejected拼接）
        policy_log_probs: 策略模型的log概率（chosen和rejected拼接）
        mask: 有效位置掩码
        beta: DPO温度参数

    返回：
        dpo损失
    """
    # 计算序列的平均log概率（忽略padding）
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask导致除零
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # -------------------- 分割chosen和rejected数据 --------------------
    batch_size = ref_log_probs.shape[0]
    # 前半是chosen，后半是rejected
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # -------------------- 计算log概率比值 --------------------
    # 策略模型：chosen vs rejected
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 参考模型：chosen vs rejected
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 优势：策略相对于参考的改进
    logits = pi_logratios - ref_logratios
    # Sigmoid损失
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    DPO训练单个epoch

    参数：
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 每个epoch的迭代数
        ref_model: 参考模型（SFT模型，冻结）
        lm_config: 模型配置
        start_step: 起始步数
        wandb: 实验追踪工具
        beta: DPO温度参数
    """
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        # -------------------- 获取数据 --------------------
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 拼接chosen和rejected数据
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # -------------------- 学习率调度 --------------------
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------------------- 计算参考模型和策略模型的log概率 --------------------
        with autocast_ctx:
            with torch.no_grad():
                # 参考模型（冻结的SFT模型）
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 策略模型（待训练的模型）
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # 计算DPO损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

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
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                  f'loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, '
                  f'aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, '
                  f'epoch_time: {eta_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # -------------------- 模型保存 --------------------
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 清理显存
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO（Direct Preference Optimization）")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（DPO通常1-2轮即可）")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size（DPO数据是成对的，所以实际token数较大）")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default=None, help="训练设备，默认为自动检测")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练（通常是full_sft）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数（控制偏离参考模型的程度）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
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
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # 参考模型（从SFT权重初始化，冻结不训练）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # 创建DPO数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
