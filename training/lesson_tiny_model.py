"""一个用于学习深度学习训练流程的极小二分类模型。

运行：
    python training/lesson_tiny_model.py

这里用两个人为设计的特征代替图片：
    第 1 列：嘴角上扬程度
    第 2 列：眉毛上扬程度

类别 0/1 只是教学用标签，不代表项目里的七种表情。
"""

import torch
from torch import nn


torch.manual_seed(42)


class TinyEmotion(nn.Module):
    """一个最小的多层感知机：2 个输入特征，2 个输出类别。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),  # 2 个输入特征 -> 8 个隐藏特征
            nn.ReLU(),        # 非线性激活
            nn.Linear(8, 2),  # 8 个隐藏特征 -> 2 个类别的 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 每行是一张“样本”，两列是两个特征。
features = torch.tensor(
    [
        [0.90, 0.80],
        [0.80, 0.70],
        [0.75, 0.90],
        [0.10, 0.20],
        [0.20, 0.10],
        [0.30, 0.25],
    ],
    dtype=torch.float32,
)

# CrossEntropyLoss 要求标签是类别编号，而不是 one-hot 向量。
labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)

model = TinyEmotion()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

for epoch in range(201):
    logits = model(features)              # 前向传播：[6, 2] -> [6, 2]
    loss = loss_function(logits, labels)  # 计算预测错误程度

    optimizer.zero_grad()                 # 清空上一步留下的梯度
    loss.backward()                       # 反向传播，计算每个参数的梯度
    optimizer.step()                      # 根据梯度更新参数

    if epoch % 50 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        print(
            f"epoch={epoch:3d} loss={loss.item():.4f} "
            f"accuracy={accuracy.item():.2f}"
        )

with torch.no_grad():
    probabilities = torch.softmax(model(features), dim=1)
    print("\n每个样本属于类别 0/1 的概率：")
    print(probabilities)
