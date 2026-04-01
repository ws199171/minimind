"""
MiniMind Web 演示界面

基于 Streamlit 的 Web 聊天界面，
支持本地模型和 API 两种调用方式。

功能特点：
- 多模型选择（MiniMind2, MiniMind2-R1, MiniMind2-MoE等）
- 对话历史管理
- 流式输出
- 推理思考过程展示（R1模型）
- 可调节参数（温度、最大生成长度等）
"""

import random
import re
from threading import Thread

import torch
import numpy as np
import streamlit as st

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# -------------------- CSS 样式 --------------------
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }
    </style>
""", unsafe_allow_html=True)

system_prompt = []
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def process_assistant_content(content):
    """
    处理助手回复内容

    将思考标签（<think>...</think>）转换为可折叠的HTML格式展示

    参数：
        content: 原始回复内容

    返回：
        处理后的HTML内容
    """
    # 检查是否R1模型
    if model_source == "API" and 'R1' not in api_model_name:
        return content
    if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    # 转换完整的思考标签
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
            content,
            flags=re.DOTALL
        )

    # 转换只有开始标签
    if '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    # 转换只有结束标签
    if '<think>' not in content and '</think>' in content:
        content = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
            content,
            flags=re.DOTALL
        )

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    """
    加载模型和分词器（带缓存）

    使用 @st.cache_resource 缓存已加载的模型，避免重复加载
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    """清空聊天消息"""
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    """初始化聊天消息显示"""
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


def regenerate_answer(index):
    """重新生成回答"""
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    """删除对话"""
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


# -------------------- 侧边栏设置 --------------------
st.sidebar.title("模型设定调整")

# 历史对话数量
st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
# 最大生成长度
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
# 温度
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)

# 模型来源
model_source = st.sidebar.radio("选择模型来源", ["本地模型", "API"], index=0)

if model_source == "API":
    # API模式配置
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minimind")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
else:
    # 本地模型配置
    MODEL_PATHS = {
        "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
        "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
        "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
        "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
        "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"]
    }

    selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=2)
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

# Logo URL
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

# -------------------- 主界面 --------------------
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    """主函数"""
    # 加载模型
    if model_source == "本地模型":
        model, tokenizer = load_model_tokenizer(model_path)
    else:
        model, tokenizer = None, None

    # 初始化消息
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # 显示历史消息
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # 聊天输入
    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")

    # 处理重新生成
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        # 显示用户消息
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        # 生成回复
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            if model_source == "API":
                # API模式
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=api_url)
                    history_num = st.session_state.history_chat_num + 1
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )

                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"API调用出错: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)
            else:
                # 本地模型模式
                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                    -(st.session_state.history_chat_num + 1):]
                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "attention_mask": inputs.attention_mask,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                Thread(target=model.generate, kwargs=generation_kwargs).start()

                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

        messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

        # 删除按钮
        with st.empty():
            if st.button("×", key=f"delete_{len(messages) - 1}"):
                st.session_state.messages = st.session_state.messages[:-2]
                st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                st.rerun()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    main()
