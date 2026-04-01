"""
Tokenizer 分词器训练脚本

本脚本用于训练 SentencePiece BPE 分词器。

注意：不建议重复训练tokenizer，MiniMind已自带预训练好的tokenizer。
基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性。
此脚本仅供学习和参考。

分词器类型：SentencePiece BPE（Byte-Pair Encoding）
- BPE 是一种常用的子词分词算法
- 将文本分割成频繁出现的子词组合
- 平衡词表大小和表示能力
"""

import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 数据路径
DATA_PATH = '../dataset/pretrain_hq.jsonl'
# 分词器保存目录
TOKENIZER_DIR = '../model_learn_tokenizer/'
# 词表大小
VOCAB_SIZE = 6400


def get_texts(data_path):
    """
    从数据文件读取文本

    参数：
        data_path: JSONL文件路径

    生成器，逐行读取并返回文本内容
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # 只使用前10000行（可调整用于测试）
                break
            data = json.loads(line)
            yield data['text']


def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    """
    训练 BPE 分词器

    参数：
        data_path: 训练数据路径
        tokenizer_dir: 分词器保存目录
        vocab_size: 词表大小
    """
    # -------------------- 初始化分词器 --------------------
    tokenizer = Tokenizer(models.BPE())

    # 字节级预处理（处理任意Unicode字符）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # -------------------- 配置训练器 --------------------
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|endoftext|>",   # 文本结束标记
            "<|im_start|>",     # 消息开始标记
            "<|im_end|>"        # 消息结束标记
        ],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # -------------------- 训练 --------------------
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # -------------------- 验证特殊token --------------------
    # 确保特殊token的ID符合预期
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # -------------------- 保存 --------------------
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # -------------------- 保存配置 --------------------
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # ChatML模板
        "chat_template": """{%- if tools %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0].role == 'system' %}
    {{- messages[0].content + '\\n\\n' }}
{%- endif %}
{{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
{%- for tool in tools %}
    {{- "\\n" }}
    {{- tool | tojson }}
{%- endfor %}
{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
 {%- if messages[0]['role'] == 'system' -%}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else -%}
        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}
 {%- endif -%}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{\\\"name\\\": \\"' }}
                {{- tool_call.name }}
                {{- '\\\", \\\"arguments\\\": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    """
    测试分词器效果

    参数：
        tokenizer_dir: 分词器目录
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 测试对话
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]

    # 应用chat模板
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-' * 100)
    print(new_prompt)

    # 测试编码和解码一致性
    print('-' * 100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")

    # 流式解码测试
    print('-' * 100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []


if __name__ == '__main__':
    # 训练分词器
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    # 测试分词器
    eval_tokenizer(TOKENIZER_DIR)
