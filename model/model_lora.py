"""
LoRA (Low-Rank Adaptation) 模型实现

LoRA 是一种高效的模型微调方法，通过在原始模型权重旁边添加低秩分解矩阵来减少训练参数量。
相比全参数微调，LoRA 可以显著降低显存占用和训练时间。

核心思想：
-冻结原始模型权重，只训练低秩矩阵A和B
- 对于输入x，输出为：h = Wx + BAx
- 其中W是原始权重，A和B是低秩矩阵

优点：
- 大幅减少可训练参数量
- 训练时不需要计算大多数参数的梯度
- 可以为不同任务训练多个LoRA权重，共享基础模型
"""

import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    LoRA 模块

    实现低秩适配的核心结构。通过两个低秩矩阵A和B的乘积来近似权重更新。

    公式：output = W @ x + B @ A @ x

    其中：
    - W: 原始权重 (in_features, out_features)，冻结不更新
    - A: 下投影矩阵 (rank, in_features)，高斯初始化
    - B: 上投影矩阵 (out_features, rank)，零初始化

    初始化策略：
    - A使用高斯分布初始化
    - B使用零初始化

    这样在训练初期，BA=0，所以输出就是原始模型的输出，保证训练的稳定性。

    参数：
        in_features: 输入特征维度（对应原始权重W的输入维度）
        out_features: 输出特征维度（对应原始权重W的输出维度）
        rank: 低秩维度，越大越接近全参数微调，但参数量也越大
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank                                     # LoRA的秩（rank），控制低秩矩阵的大小
        # 低秩矩阵A：rank × in_features
        self.A = nn.Linear(in_features, rank, bias=False)
        # 低秩矩阵B：out_features × rank
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A使用高斯分布初始化（std=0.02）
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B使用零初始化，保证训练初期BA=0
        self.B.weight.data.zero_()

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量

        返回：
            LoRA的输出 = 原始输出 + BA @ x
        """
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    为模型应用LoRA适配

    遍历模型的所有线性层，找到方阵（in_features == out_features），
    为其添加LoRA模块，并修改forward方法。

    注意：只处理方阵是因为LoRA主要应用于Attention的Q和V投影。
    其他形状的层（如K投影、O投影）通常不应用LoRA。

    参数：
        model: 需要应用LoRA的模型
        rank: LoRA的秩，默认为8
    """
    for name, module in model.named_modules():
        # 找到没有偏置的线性层，且是方阵（in_features == out_features）
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)

            # 保存原始forward方法
            original_forward = module.forward

            # 创建包装函数，显式捕获原始forward和lora模块
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 用包装函数替换原始forward
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    加载LoRA权重

    从指定路径加载LoRA权重并应用到模型。

    参数：
        model: 目标模型
        path: LoRA权重文件路径
    """
    # 加载权重文件
    state_dict = torch.load(path, map_location=model.device)

    # 处理权重名称中的"module."前缀（DDP训练时添加的）
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历模型中的LoRA模块，加载对应权重
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从state_dict中筛选出属于当前lora模块的权重
            lora_state = {k.replace(f'{name}.lora.', ''): v
                          for k, v in state_dict.items()
                          if f'{name}.lora.' in k}
            # 加载到lora模块
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    保存LoRA权重

    只保存LoRA模块的权重，不保存原始模型权重。
    这样可以高效地保存和加载不同任务的LoRA适配器。

    参数：
        model: 包含LoRA模块的模型
        path: 保存路径
    """
    # 获取原始模型（处理DDP包装）
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}

    # 遍历模型，收集所有LoRA模块的权重
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 清理名称（去除可能的"module."前缀）
            clean_name = name[7:] if name.startswith("module.") else name
            # 收集LoRA权重，添加前缀以便识别
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    # 保存到文件
    torch.save(state_dict, path)
