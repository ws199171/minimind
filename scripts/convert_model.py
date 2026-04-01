"""
模型格式转换脚本

支持以下转换：
1. PyTorch -> Transformers（MiniMind格式）
2. PyTorch -> Transformers（Llama兼容格式）
3. Transformers -> PyTorch

主要用于：
- 将训练好的PyTorch模型转换为HuggingFace格式，便于分享和部署
- 转换为Llama格式以兼容第三方生态工具
"""

import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    """
    将PyTorch模型转换为HuggingFace Transformers格式（MiniMind格式）

    参数：
        torch_path: PyTorch模型权重路径
        transformers_path: 保存目录
        dtype: 模型权重数据类型，默认为float16
    """
    # 注册自定义配置和模型到HuggingFace自动类
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # 加载模型
    lm_model = MiniMindForCausalLM(lm_config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换精度

    # 打印参数量
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')

    # 保存模型和分词器
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({
        **json.load(open(config_path, 'r', encoding='utf-8')),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {}
    }, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.float16):
    """
    将PyTorch模型转换为HuggingFace Transformers格式（Llama兼容格式）

    转换为Llama格式后，可以兼容更多第三方生态工具。

    参数：
        torch_path: PyTorch模型权重路径
        transformers_path: 保存目录
        dtype: 模型权重数据类型
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 加载权重
    state_dict = torch.load(torch_path, map_location=device)

    # 创建Llama格式配置
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True  # 共享词嵌入权重
    )

    # 创建Llama模型并加载权重
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)

    # 打印参数量
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')

    # 保存模型和分词器
    llama_model.save_pretrained(transformers_path)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # 兼容transformers-5.0的写法
    config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump({
        **json.load(open(config_path, 'r', encoding='utf-8')),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {}
    }, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(f"模型已保存为 Transformers-Llama 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """
    将HuggingFace Transformers格式转换回PyTorch格式

    参数：
        transformers_path: HuggingFace模型路径
        torch_path: 保存路径
    """
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    # 保存为半精度格式
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式 (half精度): {torch_path}")


if __name__ == '__main__':
    # 定义模型配置
    lm_config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        max_seq_len=8192,
        use_moe=False
    )

    # 定义路径
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../MiniMind2-Small'

    # 执行转换（当前示例：转换为Llama格式）
    convert_torch2transformers_llama(torch_path, transformers_path)

    # 其他转换示例：
    # convert_torch2transformers_minimind(torch_path, transformers_path)
    # convert_transformers2torch(transformers_path, torch_path)
