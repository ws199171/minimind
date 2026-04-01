"""
MiniMind PPO（Proximal Policy Optimization）训练脚本

PPO是一种基于策略的强化学习算法，用于模型对齐。
是RLHF（Reinforcement Learning with Human Feedback）的核心算法。

与DPO的区别：
- PPO需要单独的奖励模型（Reward Model）
- PPO使用广义优势估计（GAE）
- PPO有更好的理论保证和稳定性

核心组件：
1. Actor模型：待优化的策略模型
2. Critic模型：估计状态价值
3. Ref模型：参考模型，用于KL散度约束
4. Reward模型：评估回复质量

PPO核心思想：
- 使用重要性采样比率限制策略更新幅度
- 通过裁剪避免过大的策略变化
- 结合价值函数减少方差
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    """
    Critic模型

    Critic用于估计状态价值V(s)，辅助Actor优化。
    与标准语言模型的区别：输出单一价值而非词汇表分布。

    参数：
        params: 模型配置
    """

    def __init__(self, params):
        super().__init__(params)
        # 替换语言模型头为价值头
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，输出价值估计

        参数：
            input_ids: 输入token ID
            attention_mask: 注意力掩码

        返回：
            values: 价值估计
        """
        # 获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 计算价值
        values = self.value_head(hidden_states).squeeze(-1)
        return values


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
        reward_tokenizer: 奖励模型分词器

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
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if is_main_process():
            # 计算统计数据
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            # 记录日志
            if wandb is not None:
                wandb.log({
                    "actor_loss": policy_loss.item(),
                    "critic_loss": value_loss.item(),
                    "reward": rewards.mean().item(),
                    "kl": kl.item(),
                    "kl_ref": kl_ref.item(),
                    "avg_response_len": avg_len.item(),
                    "actor_lr": actor_optimizer.param_groups[0]['lr'],
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {policy_loss.item():.4f}, Critic Loss: {value_loss.item():.4f}, "
                   f"Reward: {rewards.mean().item():.4f}, KL: {kl.item():.4f}, KL_ref: {kl_ref.item():.4f}, "
                   f"Avg Response Len: {avg_len.item():.2f}")

        # 定期更新old_actor_model
        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # 保存模型
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 保存完整检查点
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model,
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default=None, help="训练设备，默认为自动检测")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="价值函数系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. 初始化模型和数据
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')

    # Old Actor模型（用于计算重要性采样比率）
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Critic模型
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    # Reward模型
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # 6. 从检查点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7. DDP包装
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

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
            ppo_train_epoch(epoch, loader, len(loader) + skip, old_actor_model, ref_model,
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model,
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)

    # 9. 清理分布式进程
    if dist.is_initialized():
        dist.destroy_process_group()
