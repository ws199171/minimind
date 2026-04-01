"""
训练工具函数集合

本模块提供训练过程中常用的工具函数，包括：
- 模型参数统计
- 分布式训练初始化
- 学习率调度
- 检查点保存/加载
- 随机种子设置
- 数据采样器等
"""

import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """
    统计并打印模型参数量

    对于MoE模型，会分别统计总参数、激活参数等信息。

    参数：
        model: 待统计的模型
        config: 模型配置
    """
    # 计算总参数量
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 获取MoE相关配置
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)

    # 计算单个专家的参数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # 计算基础参数量（不含专家）
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 计算激活参数量（每次前向传播实际参与的参数）
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 打印结果
    if active < total:
        Logger(f'Model Params: {total:.2f}M - A{active:.2f}M (激活参数)')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前是否是主进程

    在分布式训练中，只有主进程（rank=0）才执行日志打印、模型保存等操作。

    返回：
        bool: True表示是主进程
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    日志打印函数

    只在主进程打印，避免重复输出。

    参数：
        content: 日志内容
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算余弦退火学习率（带Warmup）

    学习率调度曲线：
    1. 初始阶段：lr * 0.1（warmup）
    2. 上升阶段：从 0.1lr 上升到 lr
    3. 下降阶段：按余弦曲线从 lr 下降到 0

    公式：lr * (0.1 + 0.45 * (1 + cos(pi * t/T)))

    参数：
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 最大学习率

    返回：
        当前步的学习率
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def get_device():
    """
    获取可用的计算设备

    优先级：CUDA > MPS > CPU

    返回：
        str: 设备字符串，'cuda', 'mps' 或 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_default_device():
    """
    获取默认设备字符串（带设备编号）

    返回：
        str: 设备字符串，如 'cuda:0', 'mps:0' 或 'cpu'
    """
    device = get_device()
    if device in ("cuda", "mps"):
        return f"{device}:0"
    return device


def init_distributed_mode():
    """
    初始化分布式训练模式

    支持PyTorch的DistributedDataParallel（DDP）训练。
    使用NCCL后端（GPU分布式）或Gloo后端（CPU分布式）。

    注意：MPS 设备不支持分布式训练，将自动回退到非分布式模式。

    环境变量说明：
    - RANK: 全局进程编号
    - LOCAL_RANK: 本地GPU编号
    - WORLD_SIZE: 总进程数

    返回：
        local_rank: 本地GPU编号，0表示非分布式模式
    """
    # 检查是否在分布式环境中
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非分布式模式

    # MPS 不支持分布式训练
    if torch.backends.mps.is_available():
        Logger("MPS 设备不支持分布式训练，自动回退到非分布式模式")
        return 0

    # 初始化进程组
    dist.init_process_group(backend="nccl")

    # 获取本地GPU编号
    local_rank = int(os.environ["LOCAL_RANK"])

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    """
    设置随机种子

    确保训练过程的可复现性。需要设置的随机源：
    - Python random
    - NumPy random
    - PyTorch random (CPU/GPU/MPS)
    - CuDNN deterministic (仅 CUDA)

    参数：
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                         # 多GPU随机种子
        torch.backends.cudnn.deterministic = True                # 确定性算法
        torch.backends.cudnn.benchmark = False                   # 关闭benchmark
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)                             # MPS 随机种子


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    模型检查点保存与加载

    保存内容：
    - model.state_dict(): 模型权重
    - optimizer.state_dict(): 优化器状态
    - epoch, step: 训练进度
    - world_size: GPU数量
    - wandb_id: 实验追踪ID

    参数：
        lm_config: 模型配置
        weight: 权重名称前缀
        model: 模型（None表示加载模式）
        optimizer: 优化器
        epoch: 当前轮次
        step: 当前步数
        wandb: wandb实验追踪对象
        save_dir: 保存目录
        **kwargs: 其他可保存对象（如scheduler, scaler等）

    返回：
        ckp_data: 加载的检查点数据（仅加载模式）
    """
    os.makedirs(save_dir, exist_ok=True)

    # MoE模型添加后缀
    moe_path = '_moe' if lm_config.use_moe else ''

    # 构建保存路径
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'       # 最终权重
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'  # 恢复文件

    if model is not None:
        # -------------------- 保存模式 --------------------
        # 获取原始模型（处理DDP包装）
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 获取权重
        state_dict = raw_model.state_dict()

        # 先保存为临时文件，再原子替换（防止写入中断导致文件损坏）
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取wandb run ID
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建恢复文件数据
        resume_data = {
            'model': state_dict,                               # 模型权重
            'optimizer': optimizer.state_dict(),              # 优化器状态
            'epoch': epoch,                                    # 当前轮次
            'step': step,                                       # 当前步数
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # GPU数量
            'wandb_id': wandb_id                               # wandb ID
        }

        # 添加其他对象的状态（如scheduler, critic等）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 如果对象有state_dict方法，使用它
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 保存恢复文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理显存
        del state_dict, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        # -------------------- 加载模式 --------------------
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 处理GPU数量变化的情况
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 调整step以适应新的GPU数量
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data

        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device=None):
    """
    初始化模型和分词器

    参数：
        lm_config: 模型配置
        from_weight: 加载权重的名称前缀（'none'表示不使用预训练权重）
        tokenizer_path: 分词器路径
        save_dir: 权重保存目录
        device: 设备

    返回：
        model: 初始化好的模型
        tokenizer: 分词器
    """
    # 自动检测设备
    if device is None:
        device = get_default_device()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 创建模型
    model = MiniMindForCausalLM(lm_config)

    # 加载预训练权重
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 打印模型参数统计
    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    跳过批次采样器

    用于从某个中间状态恢复训练时，跳过已经训练过的batch。

    参数：
        sampler: 原始采样器
        batch_size: 批次大小
        skip_batches: 需要跳过的batch数量
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        迭代生成批次索引

        前skip_batches个批次会被跳过。
        """
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 跳过这个batch
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        # 处理最后一个不完整的batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """返回有效批次数"""
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
