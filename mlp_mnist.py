"""
用 PyTorch 写一个多层感知机（MLP），在 MNIST 手写数字数据集上做 0~9 的十分类。

安装依赖（国内用清华镜像更快）：
    pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
运行：
    python mlp_mnist.py
第一次运行会自动下载 MNIST（约 11 MB），存到当前目录下的 data 文件夹，以后直接读取。

跑通之后可以自己改几处，再运行一次，对比测试准确率的变化：
    1. 删掉模型里的两个 nn.ReLU()
    2. 把隐藏层的 256、128 改大或改小
    3. 把 LEARNING_RATE 改成 1e-1 或 1e-5
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== 超参数 ====================
BATCH_SIZE = 64        # 每批送进模型的图片数量
LEARNING_RATE = 1e-3   # 学习率：每次更新参数时迈多大的步子
EPOCHS = 5             # 训练轮数：整个训练集完整过几遍


# ==================== 第 1 步：准备数据 ====================
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),                       # 28×28 灰度图 → 张量，像素值从 0~255 缩放到 0~1
        transforms.Normalize((0.1307,), (0.3081,)),  # 用 MNIST 的均值和标准差做标准化，训练更稳定
    ])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)   # 训练集 6 万张
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)   # 测试集 1 万张

    # DataLoader 负责把数据切成一批一批；训练集每轮打乱顺序，测试集不用打乱
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader


# ==================== 第 2 步：定义模型 ====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),          # 把 1×28×28 的图片拉平成长度 784 的向量
            nn.Linear(784, 256),   # 全连接层（隐藏层 1）：784 → 256
            nn.ReLU(),             # 非线性激活。没有它，几层线性层叠在一起仍然等价于一层线性变换
            nn.Linear(256, 128),   # 全连接层（隐藏层 2）：256 → 128
            nn.ReLU(),
            nn.Linear(128, 10),    # 输出层：10 个分数（logits），分别对应数字 0~9
        )

    def forward(self, x):
        return self.net(x)


# ==================== 第 3 步：训练一轮 ====================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                          # 切换到训练模式
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)             # 前向传播：算出每张图属于各个数字的分数
        loss = loss_fn(logits, labels)     # 计算损失：预测和真实标签差了多少

        optimizer.zero_grad()              # 先清空上一批留下的梯度（PyTorch 默认会把梯度累加起来）
        loss.backward()                    # 反向传播：自动算出每个参数的梯度
        optimizer.step()                   # 按梯度更新参数

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)   # 这一轮的平均损失


# ==================== 第 4 步：在测试集上评估 ====================
@torch.no_grad()                           # 评估时不需要梯度，关掉可以省内存、跑得更快
def evaluate(model, loader, device):
    model.eval()                           # 切换到评估模式
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)          # 分数最高的那一类就是预测结果
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)             # 准确率


# ==================== 第 5 步：把上面几步串起来 ====================
def main():
    torch.manual_seed(0)                   # 固定随机种子，每次运行的结果基本一致
    device = "cuda" if torch.cuda.is_available() else "cpu"   # 有 NVIDIA 显卡就用 GPU，没有就用 CPU
    print(f"使用设备：{device}")

    train_loader, test_loader = get_dataloaders()
    model = MLP().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{num_params:,}")

    # 交叉熵损失，分类任务的标准选择。它内部已经做了 softmax，所以模型最后一层直接输出 logits
    loss_fn = nn.CrossEntropyLoss()
    # Adam 优化器：会为每个参数自动调整步长的梯度下降方法
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"第 {epoch} 轮 | 训练损失 {train_loss:.4f} | 测试准确率 {test_acc:.2%}")

    torch.save(model.state_dict(), "mlp_mnist.pth")   # 保存训练好的参数，以后可以直接加载
    print("模型参数已保存到 mlp_mnist.pth")


if __name__ == "__main__":                 # 直接运行这个文件时，从 main() 开始执行
    main()
