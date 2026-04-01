"""
MiniMind OpenAI 兼容 API 服务

本脚本启动一个 FastAPI 服务器，提供 OpenAI 兼容的 chat completions 接口。
可以部署为 REST API 服务，供第三方应用调用。

接口格式：
- POST /v1/chat/completions

支持功能：
- 流式输出（stream=True）
- 非流式输出（stream=False）
- ChatML 模板
- 常用采样参数（temperature, top_p, max_tokens）
"""

import argparse
import json
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


def init_model(args):
    """
    初始化模型和分词器

    参数：
        args: 命令行参数

    返回：
        model: 初始化的模型
        tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)

        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    """
    Chat completions 请求格式

    与 OpenAI API 兼容的请求格式
    """
    model: str                                      # 模型名称（未使用）
    messages: list                                 # 对话消息列表
    temperature: float = 0.7                        # 生成温度
    top_p: float = 0.92                            # Nucleus采样阈值
    max_tokens: int = 8192                         # 最大生成长度
    stream: bool = False                           # 是否流式输出
    tools: list = []                              # 工具调用（暂不支持）


class CustomStreamer(TextIteratorStreamer):
    """
    自定义流式输出器

    将模型生成的内容通过队列传递给API响应
    """
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        当生成完成一段文本时调用

        参数：
            text: 生成的文本
            stream_end: 是否是最后一个文本块
        """
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def generate_stream_response(messages, temperature, top_p, max_tokens):
    """
    生成流式响应

    参数：
        messages: 对话消息列表
        temperature: 生成温度
        top_p: Nucleus采样阈值
        max_tokens: 最大生成长度

    返回：
        生成器，产生JSON格式的响应块
    """
    try:
        # 构建输入
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 创建队列和流式输出器
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        # 在新线程中启动生成
        Thread(target=_generate).start()

        # 从队列中获取并产出响应块
        while True:
            text = queue.get()
            if text is None:
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break

            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Chat completions 接口

    POST /v1/chat/completions

    参数：
        request: ChatRequest请求体

    返回：
        流式或非流式响应
    """
    try:
        if request.stream:
            # 流式响应
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # -------------------- 命令行参数 --------------------
    parser = argparse.ArgumentParser(description="MiniMind API 服务")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE外推")
    parser.add_argument('--device', default=None, type=str, help="运行设备，默认为自动检测")
    args = parser.parse_args()

    global device
    # 自动检测设备
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    # 初始化模型
    model, tokenizer = init_model(args)

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8998)
