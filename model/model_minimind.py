"""
MiniMind 核心配置文件

本文件定义了 MiniMind 模型的核心配置类，用于配置模型的各种超参数。
配置包括：模型维度、注意力头数、层数、词表大小、RoPE参数、MoE专家配置等。
"""

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类

    该类继承自 HuggingFace 的 PretrainedConfig，定义了 MiniMind 模型的所有超参数。
    包括模型架构参数、训练参数、以及可选的 MoE（混合专家）配置。

    主要参数说明：
    - hidden_size: 隐藏层维度，决定模型的宽度
    - num_hidden_layers:  Transformer 层数，决定模型的深度
    - num_attention_heads: 注意力头数，用于多头注意力机制
    - vocab_size: 词表大小，表示模型能识别的token数量
    - max_position_embeddings: 最大位置编码长度，决定模型能处理的最长序列

    MoE 相关参数（当 use_moe=True 时生效）：
    - num_experts_per_tok: 每个token选择的专家数量
    - n_routed_experts: 路由专家总数
    - n_shared_experts: 共享专家数量
    """
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            # --------------------------------------------------------
            # 以下是 MoE（混合专家）架构的专用配置
            # 当 use_moe=False 时，以下配置无效
            # --------------------------------------------------------
            use_moe: bool = False,
            num_experts_per_tok: int = 2,      # 每个token选择的专家数量
            n_routed_experts: int = 4,          # 路由专家总数
            n_shared_experts: int = 1,          # 共享专家数量
            scoring_func: str = 'softmax',      # 评分函数，默认为softmax
            aux_loss_alpha: float = 0.01,       # 辅助损失的权重系数
            seq_aux: bool = True,               # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,        # 是否对top-k概率进行归一化
            **kwargs
    ):
        super().__init__(**kwargs)
        # -------------------- 基础模型参数 --------------------
        self.dropout = dropout                                      # Dropout 概率
        self.bos_token_id = bos_token_id                            # 句子开始token的ID
        self.eos_token_id = eos_token_id                            # 句子结束token的ID
        self.hidden_act = hidden_act                                # 隐藏层激活函数类型
        self.hidden_size = hidden_size                              # 隐藏层维度
        self.intermediate_size = intermediate_size                  # FFN中间层维度
        self.max_position_embeddings = max_position_embeddings      # 最大位置编码长度
        self.num_attention_heads = num_attention_heads              # 注意力头数
        self.num_hidden_layers = num_hidden_layers                  # Transformer层数
        self.num_key_value_heads = num_key_value_heads              # Key-Value头数（用于GQA）
        self.vocab_size = vocab_size                                # 词表大小
        self.rms_norm_eps = rms_norm_eps                            # RMS Norm的epsilon
        self.rope_theta = rope_theta                                # RoPE的基础频率
        self.inference_rope_scaling = inference_rope_scaling        # 推理时是否启用RoPE缩放
        self.flash_attn = flash_attn                                # 是否使用Flash Attention加速

        # -------------------- RoPE 位置编码缩放配置 --------------------
        # 外推长度 = factor * original_max_position_embeddings = 32768
        # 使用 YaRN 方法进行位置编码外推，支持更长的上下文
        self.rope_scaling = {
            "beta_fast": 32,                                       # 快衰减beta值
            "beta_slow": 1,                                        # 慢衰减beta值
            "factor": 16,                                           # 缩放因子
            "original_max_position_embeddings": 2048,               # 原始最大位置
            "attention_factor": 1.0,                               # 注意力缩放因子
            "type": "yarn"                                         # 缩放方法类型
        } if self.inference_rope_scaling else None

        # -------------------- MoE 混合专家配置 --------------------
        # 当 use_moe=True 时启用MoE架构，多个专家网络组成混合专家系统
        self.use_moe = use_moe                                     # 是否启用MoE架构
        self.num_experts_per_tok = num_experts_per_tok             # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts                   # 路由专家总数
        self.n_shared_experts = n_shared_experts                   # 共享专家数量
        self.scoring_func = scoring_func                            # 评分函数（softmax/sigmoid等）
        self.aux_loss_alpha = aux_loss_alpha                        # 辅助损失的alpha参数
        self.seq_aux = seq_aux                                     # 是否在序列级别计算辅助损失
        self.norm_topk_prob = norm_topk_prob                        # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model 核心实现
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）层

    RMSNorm 是 LayerNorm 的简化版本，只使用 RMS（均方根）进行归一化，
    相比 LayerNorm 减少了计算量，同时在许多任务上效果相当。

    公式：output = weight * (x / RMS(x)) * eps
    其中 RMS(x) = sqrt(mean(x^2))

    参数：
        dim: 输入张量的维度
        eps: 防止除零的小常数，默认为 1e-5
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps                                           # 归一化epsilon，防止除零
        self.weight = nn.Parameter(torch.ones(dim))             # 可学习的缩放参数gamma

    def _norm(self, x):
        """
        执行 RMS 归一化

        计算 x 的均方根，然后返回归一化后的值
        rsqrt 是平方根的倒数，即 1/sqrt(x)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量，形状为 (batch, seq_len, hidden_size)

        返回：
            归一化后的张量，形状与输入相同
        """
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算旋转位置编码（RoPE）的cos和sin频率

    旋转位置编码（RoPE）是一种将位置信息融入注意力机制的方法。
    它通过将Token的Query和Key向量旋转相应的角度来编码位置信息。

    参数：
        dim: 注意力头的维度（head_dim）
        end: 最大位置长度
        rope_base: RoPE的基础频率，默认为1e6（用于长上下文）
        rope_scaling: 位置编码缩放配置（用于外推）

    返回：
        freqs_cos: cos预计算结果
        freqs_sin: sin预计算结果

    YaRN 缩放原理：
        f'(i) = f(i) * ((1-gamma) + gamma/s)
        其中 gamma 是线性衰减的ramp函数，s 是缩放因子
    """
    # 计算基础频率：rope_base^(-2i/dim)，i从0到dim/2
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # 如果提供了rope_scaling配置，使用YaRN方法进行位置编码外推
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), 其中 γ∈[0,1] 是线性ramp函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 创建从0到1的线性ramp
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 应用YaRN缩放
            freqs = freqs * (1 - ramp + ramp / factor)

    # 创建位置索引
    t = torch.arange(end, device=freqs.device)
    # 计算每对位置的角度（外积）
    freqs = torch.outer(t, freqs).float()
    # 计算cos和sin，cat后维度变为dim
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到Query和Key向量

    这是RoPE的核心实现。通过将Query和Key旋转特定角度来融入位置信息。

    旋转公式：
        q' = q * cos(θ) + rotate_half(q) * sin(θ)
        k' = k * cos(θ) + rotate_half(k) * sin(θ)

    参数：
        q: Query向量，形状为 (batch, seq_len, num_heads, head_dim)
        k: Key向量，形状与q类似
        cos: 预计算的cos值
        sin: 预计算的sin值
        position_ids: 位置ID（可选）
        unsqueeze_dim: 需要扩展的维度，默认为1

    返回：
        q_embed: 应用RoPE后的Query
        k_embed: 应用RoPE后的Key
    """

    def rotate_half(x):
        """
        将向量按维度旋转180度（将后半部分放到前面）

        用于RoPE计算：x = [x1, x2, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
        rotate_half(x) = [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转：原始向量乘以cos + 旋转后的向量乘以sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将Key-Value向量重复扩展以匹配Query数量

    用于Grouped Query Attention（GQA）中，当Key-Value头数少于Query头数时，
    需要将Key-Value向量复制扩展。

    参数：
        x: 输入张量，形状为 (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 重复次数 = num_query_heads / num_kv_heads

    返回：
        扩展后的张量，形状为 (batch, seq_len, num_query_heads, head_dim)

    示例：
        如果有8个Query头和2个KV头，则n_rep=4
        每个KV头需要复制4次来匹配8个Query头
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 使用expand和reshape实现高效复制
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制模块

    实现了标准的自注意力机制，支持：
    - Grouped Query Attention (GQA)：减少KV头数，降低内存占用
    - Flash Attention：高效的注意力计算
    - RoPE 旋转位置编码：融入位置信息
    - KV Cache：加速推理

    参数：
        args: MiniMindConfig配置对象
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # GQA 配置：Key-Value头数可以少于Query头数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0, "Query头数必须能被KV头数整除"
        self.n_local_heads = args.num_attention_heads                              # 本地Query头数
        self.n_local_kv_heads = self.num_key_value_heads                          # 本地KV头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads                  # 每个KV头需要复制的次数
        self.head_dim = args.hidden_size // args.num_attention_heads              # 每个头的维度

        # QKV 投影层：将hidden_size映射到Query/Key/Value空间
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # Output投影：将多头注意力的输出映射回hidden_size
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)                              # 注意力dropout
        self.resid_dropout = nn.Dropout(args.dropout)                             # 残差dropout
        self.dropout = args.dropout

        # Flash Attention 配置：检查是否可用
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) 元组
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播

        参数：
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            position_embeddings: 位置编码的(cos, sin)元组
            past_key_value: 之前的KV cache，用于加速推理
            use_cache: 是否返回KV cache
            attention_mask: 注意力掩码，用于处理padding

        返回：
            output: 注意力输出
            past_kv: 更新后的KV cache（如果use_cache=True）
        """
        bsz, seq_len, _ = x.shape

        # -------------------- QKV 投影 --------------------
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 重塑为多头格式：(batch, seq_len, num_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # -------------------- 应用 RoPE 位置编码 --------------------
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # -------------------- KV Cache 实现 --------------------
        # 推理时，将之前的KV与当前的KV拼接，避免重复计算
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # -------------------- 多头格式转换 --------------------
        # 对KV进行复制扩展以匹配Query头数（用于GQA）
        xq, xk, xv = (
            xq.transpose(1, 2),                                        # (batch, heads, seq, dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2),                 # (batch, heads, seq, dim)
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # -------------------- 注意力计算 --------------------
        # 优先使用Flash Attention（更高效）
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention：自动处理因果掩码和注意力计算
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 标准注意力计算（兼容性更强）
            # 计算注意力分数：Q @ K^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 应用因果掩码（causal mask）- 防止看到未来信息
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 添加注意力掩码（如padding掩码）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax归一化
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 注意力加权求和
            output = scores @ xv

        # -------------------- 输出重塑和投影 --------------------
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # 合并多头
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络（FFN）模块

    也称为MLP，是Transformer中每个Block的重要组成部分。
    采用Swish/Gated SiLU激活函数：
        FFN(x) = down_proj(act_fn(gate_proj(x)) * up_proj(x))

    参数：
        config: MiniMindConfig配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 计算中间层维度，确保是64的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # 门控投影：hidden -> intermediate
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 上投影：intermediate -> hidden
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 下投影：hidden -> intermediate
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)                            # Dropout层
        self.act_fn = ACT2FN[config.hidden_act]                              # 激活函数

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量，形状为 (batch, seq_len, hidden_size)

        返回：
            输出张量，形状与输入相同
        """
        # SiLU/Swish激活：x * sigmoid(x)
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    混合专家系统的门控网络

    MoE（Mixture of Experts）通过门控机制选择最适合处理当前输入的专家。
    每个输入token会被路由到top-k个专家进行加权处理。

    参数：
        config: MiniMindConfig配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok                      # 每个token选择的专家数
        self.n_routed_experts = config.n_routed_experts               # 专家总数

        self.scoring_func = config.scoring_func                       # 评分函数
        self.alpha = config.aux_loss_alpha                            # 辅助损失权重
        self.seq_aux = config.seq_aux                                 # 序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob                  # 是否标准化top-k概率
        self.gating_dim = config.hidden_size                          # 门控输入维度
        # 门控权重矩阵：(num_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化门控权重
        使用Kaiming均匀分布初始化
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        门控前向传播：计算每个专家的权重并选择top-k

        参数：
            hidden_states: 输入张量，形状为 (batch, seq_len, hidden_size)

        返回：
            topk_idx: 选中的专家索引
            topk_weight: 选中专家的权重（归一化后）
            aux_loss: 辅助损失（用于负载均衡）
        """
        bsz, seq_len, h = hidden_states.shape
        # 展平为2D：(batch*seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, h)

        # -------------------- 门控评分 --------------------
        # 计算每个专家的logits：hidden_states @ weight^T
        logits = F.linear(hidden_states, self.weight, None)

        # 计算概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的MoE评分函数: {self.scoring_func}')

        # -------------------- Top-K 选择 --------------------
        # 选择概率最高的top-k个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # -------------------- 概率归一化 --------------------
        # 如果选择多个专家，对权重进行归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # -------------------- 辅助损失（负载均衡） --------------------
        # 鼓励专家被均匀选择，防止某些专家被过度使用或几乎不使用
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 序列级别辅助损失：在序列维度上计算
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 统计每个专家被选择的频率
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # 辅助损失 = 专家选择频率 * 专家分数
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token级别辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)                                      # 选择频率
                Pi = scores_for_aux.mean(0)                                      # 平均概率
                fi = ce * self.n_routed_experts                                   # 归一化频率
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络（MoE FFN）

    由多个独立的FFN专家网络和一个门控网络组成。
    每个输入token会被路由到最合适的top-k个专家处理。

    参数：
        config: MiniMindConfig配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建多个独立的FFN专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络：决定每个token由哪些专家处理
        self.gate = MoEGate(config)

        # 共享专家：始终参与计算，不经过门控
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        MoE前向传播

        参数：
            x: 输入张量，形状为 (batch, seq_len, hidden_size)

        返回：
            混合专家的输出
        """
        identity = x                                        # 残差连接的原输入
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # -------------------- 门控选择 --------------------
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 展平处理
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        # -------------------- 专家计算 --------------------
        if self.training:
            # 训练模式：复制token以并行计算多个专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            # 为每个专家分配其负责的token并计算
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 处理空expert输出的情况（保持计算图连通）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())

            # 加权合并多个专家的输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式：优化每个专家的token分组计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # -------------------- 共享专家 --------------------
        # 共享专家的处理结果直接加到输出上
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss                              # 保存辅助损失用于反向传播
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE推理模式的高效实现

        通过对token按专家分组，减少了复制开销，提高了推理效率。

        参数：
            x: 展平后的输入 (batch*seq_len, hidden_size)
            flat_expert_indices: 每个token分配的专家索引
            flat_expert_weights: 每个token的专家权重

        工作原理：
            假设 experts_indices = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...]
            经过排序和分组后，同一个专家处理的token被连续排列，
            这样可以一次性处理多个相同专家负责的token。
        """
        expert_cache = torch.zeros_like(x)
        # 按专家索引排序，便于分组处理
        idxs = flat_expert_indices.argsort()
        # 统计每个专家处理的token数量（累积和）
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 计算每个token属于哪个位置（用于scatter_add）
        token_idxs = idxs // self.config.num_experts_per_tok

        # -------------------- 分组处理每个专家 --------------------
        # tokens_per_expert = [6, 15, 20, 26] 表示：
        # - 专家0处理 token_idxs[:6]
        # - 专家1处理 token_idxs[6:15]
        # - 专家2处理 token_idxs[15:20]
        # - 专家3处理 token_idxs[20:26]
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[i]                           # 获取专家模型
            exp_token_idx = token_idxs[start_idx:end_idx]      # 该专家负责的token索引
            expert_tokens = x[exp_token_idx]                   # 提取token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 专家计算

            # 乘以专家权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到缓存中（使用scatter_add处理一个token被多个专家处理的情况）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer Block（层）

    每个Block包含：
    1. RMSNorm（输入归一化）
    2. 自注意力层（Self-Attention）
    3. RMSNorm（注意力后归一化）
    4. 前馈网络（FFN）或 MoE FFN

    参数：
        layer_id: 当前层的索引
        config: MiniMindConfig配置对象
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 自注意力层
        self.self_attn = Attention(config)

        self.layer_id = layer_id                               # 当前层编号

        # Pre-attention RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Post-attention RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 前馈网络：根据配置选择标准FFN或MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播

        参数：
            hidden_states: 输入隐藏状态
            position_embeddings: RoPE位置编码
            past_key_value: KV cache
            use_cache: 是否返回KV cache
            attention_mask: 注意力掩码

        返回：
            hidden_states: 输出隐藏状态
            present_key_value: 更新后的KV cache
        """
        # 残差连接
        residual = hidden_states

        # -------------------- Pre-Norm 结构 --------------------
        # 与其使用 Post-Norm (输出 += Norm(x)), 不如使用 Pre-Norm: x + Sublayer(Norm(x))
        # Pre-Norm 在训练初期更加稳定

        # 自注意力 + 残差连接
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual                              # 残差连接

        # 前馈网络 + 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 基础模型（不含语言模型头）

    包含：
    - 词嵌入层
    - Transformer Blocks堆叠
    - 最终RMSNorm
    - RoPE位置编码

    参数：
        config: MiniMindConfig配置对象
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入层：将token ID映射为向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer Blocks堆叠
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # -------------------- 预计算 RoPE 位置编码 --------------------
        # 为所有可能的位置预计算cos和sin值，加速推理
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册为buffer（非可学习参数，但会随模型保存/加载）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        前向传播

        参数：
            input_ids: 输入token ID，形状为 (batch, seq_len)
            attention_mask: 注意力掩码
            past_key_values: KV cache列表
            use_cache: 是否返回KV cache

        返回：
            hidden_states: 输出隐藏状态
            presents: KV cache列表
            aux_loss: MoE辅助损失
        """
        batch_size, seq_length = input_ids.shape

        # 如果past_key_values是DDP格式，设为None
        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        # 初始化past_key_values列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算起始位置（用于从缓存位置继续）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # -------------------- 词嵌入 + Dropout --------------------
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # -------------------- 获取位置编码 --------------------
        # 根据当前序列长度动态获取对应的RoPE编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # -------------------- 通过Transformer Blocks --------------------
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # -------------------- 最终归一化 --------------------
        hidden_states = self.norm(hidden_states)

        # -------------------- 汇总MoE辅助损失 --------------------
        # 累加所有MoE层的辅助损失
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型

    在 MiniMindModel 基础上添加了语言模型头（LM Head），
    用于根据隐藏状态预测下一个token。

    参数：
        config: MiniMindConfig配置对象
    """

    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        # 使用提供的配置或默认配置
        self.config = config or MiniMindConfig()
        super().__init__(self.config)

        # 基础模型
        self.model = MiniMindModel(self.config)

        # 语言模型头：将隐藏状态映射为词表概率分布
        # 注意：这里共享了embed_tokens的权重（权重绑定），减少参数量
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播

        参数：
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            past_key_values: KV cache
            use_cache: 是否返回KV cache
            logits_to_keep: 保留的logits数量（用于加速）

        返回：
            CausalLMOutputWithPast对象，包含loss、logits等
        """
        # -------------------- 基础模型前向传播 --------------------
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # -------------------- 语言模型头 --------------------
        # 只计算最后logits_to_keep个位置的logits（加速用）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # -------------------- 计算损失 --------------------
        loss = None
        if labels is not None:
            # 因果语言模型：预测下一个token
            # 将logits和labels错开一位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熵损失，忽略padding位置的loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # -------------------- 封装输出 --------------------
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
        # 添加MoE辅助损失
        output.aux_loss = aux_loss

        return output
