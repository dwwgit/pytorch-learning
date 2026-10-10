# 导入torch
import torch

# 建立两个张量
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])
#输出x，y
print(x)
print(y)

#建立参数w,同时告诉pytorch之后要对他求梯度，打印w
w = torch.tensor(0.0,requires_grad=True)
print(w)

#建优化器
optimizer = torch.optim.SGD([w], lr=0.001)

#算预测值和损失
pred = w * x
print(pred)
loss = ((y-pred) ** 2).mean()
print(loss)

#反向传播，看梯度
loss.backward()
print(w.grad)

#更新参数
optimizer.step()
print(w)

#加循环,把预测到更新参数这几行循环了
for step in range(30):
    pred = w * x
    loss = ((y-pred) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(step, loss.item(), w.item())