#导入库
import torch
from torch.autograd import grad

from step2 import pred

#设置x，y张量,并输出
x = torch.tensor([1.0,2.0,3.0,4.0])
y = torch.tensor([2.0,4.0,6.0,8.0])
print(x)
print(y)

#设置w初始值为0，同时标记后来求梯度,并输出
w = torch.tensor(0.0,requires_grad=True)
print(w)

#计算预测和损失
pred = w * x
print(pred)
loss = ((y - pred) ** 2).mean()
print(loss)

#反向传播，看梯度
loss.backward()
print(w.grad)


