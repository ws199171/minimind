"""
MiniMind 预训练脚本

预训练阶段使用大规模无标注文本数据，通过自回归语言建模任务训练基础语言模型。
预训练让模型学习语言的基本规律、语法、语义和世界知识。

训练特点：
- 全参数训练：所有模型参数都会更新
- 大批量：通常使用较大的batch size
- 长训练：需要训练多个epoch
- 高学习率：相比微调使用更高的学习率
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
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,                    # 余弦退火学习率计算
    Logger,                    # 日志打印
    is_main_process,           # 判断主进程
    lm_checkpoint,             # 检查点保存/加载
    init_distributed_mode,     # 分布式训练初始化
    setup_seed,                # 随机种子设置
    init_model,                # 模型初始化
    SkipBatchSampler           # 跳过批次采样器
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    预训练单个epoch

    参数：
        epoch: 当前epoch编号
        loader: 数据加载器
        iters: 每个epoch的迭代数
        start_step: 起始步数（用于恢复训练）
        wandb: 实验追踪工具
    """
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # -------------------- 数据准备 --------------------
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # -------------------- 学习率调度 --------------------
        # 使用余弦退火学习率，配合warmup
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # -------------------- 混合精度前向传播 --------------------
        with autocast_ctx:
            # 模型前向传播，计算loss
            res = model(input_ids, labels=labels)
            # 总loss = 语言模型loss + MoE辅助loss
            loss = res.loss + res.aux_loss
            # 梯度累积：将多个batch的loss累加后再反向传播
            loss = loss / args.accumulation_steps

        # -------------------- 反向传播 --------------------
        scaler.scale(loss).backward()

        # -------------------- 梯度更新 --------------------
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度的scaler缩放
            scaler.unscale_(optimizer)
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度（使用set_to_none更高效）
            optimizer.zero_grad(set_to_none=True)

        # -------------------- 日志输出 --------------------
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss       # 纯语言模型loss
            current_lr = optimizer.param_groups[-1]['lr']

            # 估算剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                  f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                  f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                  f'epoch_time: {eta_min:.1f}min')

            # wandb记录
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

            # 保存最终权重
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 保存完整检查点（包含优化器状态，用于恢复训练）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                         optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                         wandb=wandb, save_dir='../checkpoints')

            model.train()
            del state_dict

        # 清理显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # -------------------- 命令行参数解析 --------------------
    parser = argparse.ArgumentParser(description="MiniMind 预训练")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="分词器路径")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default=None, help="训练设备，默认为自动检测")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
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
    # 设置随机种子，确保可复现性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    # 检查是否有可恢复的检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else ("mps" if "mps" in args.device else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # MPS 不支持 float16 的 GradScaler，建议使用 bfloat16
    if device_type == "mps" and args.dtype == "float16":
        Logger("MPS 设备不支持 float16 混合精度，自动切换为 bfloat16")
        dtype = torch.bfloat16
    # 混合精度上下文：CPU训练不使用自动混合精度
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配置Wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, device=args.device)

    # torch.compile加速（PyTorch 2.0+特性）
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 创建预训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 分布式采样器（确保各GPU数据不重复）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 梯度缩放器（用于混合精度训练，CUDA 使用 GradScaler，MPS/CPU 不需要）
    use_grad_scaler = device_type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=use_grad_scaler)

    # AdamW优化器
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
        # 忽略freqs_cos和freqs_sin buffer（不需要同步）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置epoch（用于分布式训练的数据 shuffle）
        train_sampler and train_sampler.set_epoch(epoch)

        # 设置随机种子（每个epoch使用不同的shuffle）
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        # 计算跳过的batch数（用于恢复训练）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # 创建batch采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
