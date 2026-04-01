"""
MiniMind 模型推理与评测脚本

支持两种模式：
1. 自动测试：使用预设问题测试模型能力
2. 手动输入：用户交互式对话

支持的评测基准：
- C-Eval：中文综合能力评测
- CMMLU：中文多任务语言理解
- MMLU：英文多任务语言理解
- GSM8K：数学推理评测
- MBPP：编程能力评测
"""

import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args):
    """
    初始化模型和分词器

    支持两种加载方式：
    1. 从原生PyTorch格式加载（load_from='model'）
    2. 从HuggingFace格式加载（其他路径）

    参数：
        args: 命令行参数，包含模型路径、配置等

    返回：
        model: 初始化好的模型
        tokenizer: 分词器
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from:
        # -------------------- 原生PyTorch格式加载 --------------------
        # 创建模型配置
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))

        # 构建权重路径
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # 加载权重
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # 加载LoRA权重（如指定）
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # -------------------- HuggingFace格式加载 --------------------
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    # 打印模型参数
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    """
    主函数：模型推理与对话
    """
    # -------------------- 命令行参数解析 --------------------
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default=None, type=str, help="运行设备，默认为自动检测")
    args = parser.parse_args()

    # 自动检测设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # -------------------- 预设测试问题 --------------------
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = []  # 对话历史

    # 初始化模型
    model, tokenizer = init_model(args)

    # 选择输入模式
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # 创建流式输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 选择问题来源
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')

    for prompt in prompt_iter:
        # 设置随机种子（确保可复现）
        setup_seed(2026)

        # 打印用户输入
        if input_mode == 0:
            print(f'💬: {prompt}')

        # 更新对话历史（限制长度）
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 构建输入模板
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        # Reason模型启用思考标签
        if args.weight == 'reason':
            templates["enable_thinking"] = True

        # Tokenize输入
        if args.weight != 'pretrain':
            inputs = tokenizer.apply_chat_template(**templates)
        else:
            inputs = tokenizer.bos_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 生成回复
        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0
        )

        # 解码回复
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        # 更新对话历史
        conversation.append({"role": "assistant", "content": response})

        # 计算并显示生成速度
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')


if __name__ == "__main__":
    main()
