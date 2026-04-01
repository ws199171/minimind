"""
数据集处理模块

本模块定义了各种训练任务所需的数据集类：
- PretrainDataset: 预训练数据集
- SFTDataset: 监督微调数据集
- DPODataset: DPO偏好优化数据集
- RLAIFDataset: 强化学习人类反馈数据集

每个数据集类都继承自PyTorch的Dataset，用于提供标准化的数据接口。
"""

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话预处理：随机添加系统提示

    为了增加数据多样性，以一定概率在对话开头添加系统提示。

    参数：
        conversations: 对话列表，每项为 {"role": str, "content": str}
        add_system_ratio: 添加系统提示的概率，默认为0.2（20%）

    返回：
        添加系统提示后的对话列表
    """
    # 预定义多种系统提示（中英文混合）
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI Assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    # 检查是否需要添加系统提示
    # 条件：对话不为空，且第一条不是系统消息，且随机值小于阈值
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            # 从预定义列表中随机选择一个系统提示
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations

    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：随机移除空的思考标签

    用于训练推理模型时，以一定概率移除空的思考标签。
    这样可以让模型学会处理有/无思考的情况。

    参数：
        prompt_content: 处理后的提示内容
        empty_think_ratio: 移除空思考标签的概率，默认为0.05

    返回：
        处理后的内容
    """
    # 检查是否包含空的思考标签
    if '<think>\n\n</think>\n\n' in prompt_content:
        # 以一定概率保留空思考标签
        if random.random() > empty_think_ratio:
            # 移除空的思考标签
            prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')

    return prompt_content


class PretrainDataset(Dataset):
    """
    预训练数据集

    用于语言模型的预训练阶段。数据格式为纯文本，
    通过自回归方式预测下一个token。

    数据处理流程：
    1. 将文本tokenize
    2. 添加BOS和EOS特殊token
    3. padding到固定长度
    4. labels设为-100用于padding位置（不计算loss）
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集

        参数：
            data_path: 数据文件路径（JSONL格式）
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用HuggingFace datasets库加载数据
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个样本

        返回：
            input_ids: 输入token ID序列
            labels: 标签序列（用于计算loss）
        """
        sample = self.samples[index]

        # Tokenize文本，设置最大长度（预留BOS和EOS的位置）
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 添加BOS（句子开始）和EOS（句子结束）特殊token
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Padding到固定长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 创建labels：复制input_ids，但padding位置设为-100（不计算loss）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


class SFTDataset(Dataset):
    """
    监督微调（Supervised Fine-Tuning）数据集

    用于SFT阶段，数据格式为多轮对话。
    只计算assistant回复部分的loss，不计算user输入的loss。

    这是一种非对称的注意力掩码机制：
    - User输入对整个序列可见
    - Assistant输出只对自身和之前的token可见
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化SFT数据集

        参数：
            jsonl_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        # 预计算assistant和EOS的token ID（包含换行符）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        创建聊天提示模板

        使用tokenizer的apply_chat_template方法将对话列表转换为特定格式。
        支持带有工具调用（functions）的对话格式。

        参数：
            conversations: 对话列表

        返回：
            格式化后的提示字符串
        """
        messages = conversations.copy()

        # 检查是否包含工具调用
        tools = conversations[0]["functions"] if (
            conversations and
            conversations[0]["role"] == "system" and
            conversations[0].get("functions")
        ) else None

        # 使用tokenizer的chat template
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        生成标签掩码

        只在assistant回复部分计算loss，user输入部分的labels设为-100。

        实现逻辑：
        1. 找到所有assistant回复的起始位置（通过bos_id定位）
        2. 找到对应的EOS位置
        3. 标记assistant回复部分的labels为原始token ID

        参数：
            input_ids: 输入token ID序列

        返回：
            labels: 标签序列
        """
        labels = [-100] * len(input_ids)                   # 初始化所有位置为-100
        i = 0

        while i < len(input_ids):
            # 检查当前位置是否匹配assistant起始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # assistant回复的起始位置
                start = i + len(self.bos_id)
                end = start

                # 找到assistant回复的结束位置（EOS标记）
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 标记assistant回复部分的labels（包含EOS）
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]

                # 移动到下一个位置（跳过整个assistant回复）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return labels

    def __getitem__(self, index):
        """
        获取单个样本

        返回：
            input_ids: 输入token ID序列
            labels: 标签序列（只有assistant部分有值）
        """
        sample = self.samples[index]

        # 对话预处理：可能添加系统提示
        conversations = pre_processing_chat(sample['conversations'])

        # 创建chat prompt
        prompt = self.create_chat_prompt(conversations)

        # 后处理：可能移除空思考标签
        prompt = post_processing_chat(prompt)

        # Tokenize并截断
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # Padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成labels
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """
    DPO（Direct Preference Optimization）偏好优化数据集

    DPO是一种基于偏好的训练方法，数据包含：
    - chosen: 偏好回复（更好的回答）
    - rejected: 拒绝回复（更差的回答）

    训练目标：使模型倾向于选择chosen而非rejected。
    """

    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化DPO数据集

        参数：
            file_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 处理padding token ID
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 预计算token ID
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

        # 加载数据
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个样本

        返回：
            包含chosen和rejected数据的字典：
            - x_chosen/y_chosen/mask_chosen: chosen序列的输入/labels/loss掩码
            - x_rejected/y_rejected/mask_rejected: rejected序列的输入/labels/loss掩码
        """
        sample = self.samples[index]

        # 获取chosen和rejected回复
        chosen = sample['chosen']                              # 偏好回复
        rejected = sample['rejected']                          # 拒绝回复

        # 处理chosen：应用chat template
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        # 处理rejected：应用chat template
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Tokenize并padding
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )

        # 提取数据
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # 错位生成输入和标签（用于语言模型训练）
        # x = input[:-1], y = input[1:]
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,                              # chosen输入
            'y_chosen': y_chosen,                              # chosen标签
            'mask_chosen': mask_chosen,                        # chosen loss掩码
            'x_rejected': x_rejected,                          # rejected输入
            'y_rejected': y_rejected,                          # rejected标签
            'mask_rejected': mask_rejected                     # rejected loss掩码
        }

    def generate_loss_mask(self, input_ids):
        """
        生成loss掩码

        与SFTDataset类似，只在assistant回复部分计算loss。

        参数：
            input_ids: 输入token ID序列

        返回：
            loss_mask: 0/1掩码，1表示计算loss
        """
        loss_mask = [0] * len(input_ids)
        i = 0

        while i < len(input_ids):
            # 找到assistant起始位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                # 找到assistant结束位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 标记assistant回复部分
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return loss_mask


class RLAIFDataset(Dataset):
    """
    RLAIF（Reinforcement Learning with AI Feedback）数据集

    用于强化学习训练的数据集。数据格式为多轮对话，
    返回prompt和answer，用于RL训练中的reward计算。

    特点：
    - 只返回prompt（不含answer的完整回复）
    - answer单独返回用于reward计算
    - 不需要生成labels
    """

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化RLAIF数据集

        参数：
            jsonl_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

        # 预计算token ID（不带换行符）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        创建聊天提示

        将对话列表转换为prompt格式，添加generation_prompt=True
        表示模型需要继续生成回复。

        参数：
            conversations: 对话列表

        返回：
            prompt: 处理后的提示
            answer: 最后一轮assistant的回复
        """
        messages = []
        answer = ''

        # 解析对话：奇数位是user，偶数位是assistant
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']

        # 应用chat template，添加generation prompt
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],                                     # 排除最后一轮（用于生成）
            tokenize=False,
            add_generation_prompt=True                        # 关键：添加assistant开始标记
        )

        # 后处理
        prompt = post_processing_chat(prompt)

        return prompt, answer

    def __getitem__(self, index):
        """
        获取单个样本

        返回：
            prompt: 输入提示（不包含assistant回复）
            answer: assistant应该生成的回复
        """
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,                                  # 用于模型输入
            'answer': answer                                   # 用于reward计算
        }


if __name__ == "__main__":
    # 测试用代码
    pass
