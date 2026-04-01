"""
MiniMind GRPO（Group Relative Policy Optimization）训练脚本

GRPO是一种无参考模型的强化学习算法，是PPO的简化版本。

与PPO的区别：
- 不需要单独的参考模型（ref_model）
- 使用组内相对优势估计
- 样本效率更高

GRPO核心思想：
- 对每个prompt生成多个回复（group）
- 计算组内每个回复的相对优势
- 优势 = (回复reward - 组内平均reward) / 组内标准差
- 优化目标：增加高优势回复的概率，降低低优势回复的概率

优势：
- 减少显存占用（不需要ref_model）
- 训练更稳定
- 超参数更少
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    计算奖励分数

    整合多个奖励源：
    1. 格式奖励：推理模型的思考标签格式
    2. Reward模型评分：外部奖励模型的评分

    参数：
        prompts: prompt列表
        responses: 回复列表
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型的分词器

    返回：
        rewards: 每个回复的奖励分数
    """
    def reasoning_model_reward(rewards):
        """
        推理格式奖励

        检查回复是否符合思考标签格式：
        -<think>...</think>：思考过程
        -<answer>...</answer>：最终答案
        """
        # 匹配标准格式
        pattern = r"^<think>\n.*?\n</think>\n'total_tokens

        loss = policy_loss + aux_loss
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {policy_loss_val:.4f}, reward: {avg_reward_val:.4f}')

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 保存模型...

        del prompt_inputs, outputs, per_token_logps, ref_per_token_logps, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    Args:
        epoch: 当前训练轮次
        loader: 数据加载器
        iters: 总迭代次数
        ref_model: 参考模型（用于计算KL散度）
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型分词器
        start_step: 起始步数（用于恢复训练）
        wandb: wandb日志记录器
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']

        # Tokenize prompts
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 生成多个回复（group）
        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id
            )

        # 提取回复部分
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]

        # 计算每个token的log概率
        def get_per_token_logps(mdl, input_ids, n_keep):
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)

        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # 解码回复文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 计算奖励
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)

        # -------------------- GRPO核心：组内相对优势 --------------------
        # 将奖励按组reshape
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        # 计算优势（标准化）
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------------------- 创建掩码 --------------------
        # EOS掩码：标记有效回复长度
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        # -------------------- 计算策略损失 --------------------
        # KL散度
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # GRPO损失：-E[exp(log_prob * advantage)]
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    # ... (参数定义与train_ppo.py类似)

    # ========== 训练循环 ==========
    # 1. 初始化环境和随机种子
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

    # 2. 配置目录、模型参数、检查点
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # 3. 设置混合精度
    device_type = "cuda" if "cuda" in args.device else ("mps" if "mps" in args.device else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "mps" and args.dtype == "float16":
        Logger("MPS 设备不支持 float16 混合精度，自动切换为 bfloat16")
        dtype = torch.bfloat16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # 4. 配置Wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 初始化模型和数据
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)

    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Reward模型
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # 6. 从检查点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7. DDP包装
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)

    # 9. 清理分布式进程
    if dist.is_initialized():
        dist.destroy_process_group()
