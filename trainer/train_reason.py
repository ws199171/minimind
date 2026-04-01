"""
MiniMind 推理能力蒸馏训练脚本

本脚本用于训练模型的推理能力，特别是CoT（Chain of Thought）思维链能力。

核心特点：
- 使用带有推理过程的训练数据（如R1格式）
- 特殊token标记：<think>...</think> 包裹思考过程，<answer>...<answer> 包裹最终答案
- 对特殊token给予更高权重（10倍），使模型更注重格式

推理格式：
   <think>
    思考过程...
   </think>
    <answer>
    最终答案
    </answer>
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    """
    推理蒸馏训练单个epoch

    与标准SFT的区别：
    1. 对特殊token（思考标签、答案标签）给予更高loss权重
    2. 确保模型学会遵循推理格式

    参数：
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 每个epoch的迭代数
        tokenizer: 分词器（用于获取特殊token ID）
        lm_config: 模型配置
        start_step: 起始步数
        wandb: 实验追踪工具
    """
    start_time = time.time()

    # -------------------- 预计算特殊token ID --------------------
    # 这些token用于标记思考过程和答案
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids

    # 损失函数（不进行reduction，手动计算）
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # -------------------- 数据准备 --------------------
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # -------------------- 学习率调度 --------------------
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------------------- 混合精度前向传播 --------------------
        with autocast_ctx:
            res = model(input_ids)
            # 预测下一个token的logits
            shift_logits = res.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算每个位置的loss
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())

            # -------------------- 创建loss掩码 --------------------
            # 基础掩码：忽略-100的位置
            loss_mask = (shift_labels != -100).float()

            # 标记特殊token位置
            sp_ids = torch.isin(
                shift_labels.view(-1),
                torch.tensor(
                    start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids
                ).to(args.device)
            )

            loss_mask_flat = loss_mask.view(-1)
            loss_mask_sum = loss_mask_flat.sum()

            # 对特殊token给予10倍权重
            loss_mask_flat[sp_ids] = 10
            loss_mask = loss_mask_flat.view(shift_labels.size())

            # 计算加权loss
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum

            # 加上MoE辅助损失
            loss = logits_loss + res.aux_loss
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
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                  f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                  f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                  f'epoch_time: {eta_min:.1f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
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

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 推理能力蒸馏")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default=None, help="训练设备，默认为自动检测")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练（通常为dpo）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

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
            train_epoch(epoch, loader, len(loader) + skip, tokenizer, lm_config, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
