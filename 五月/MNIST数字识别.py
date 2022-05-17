import torch

from model import MLP
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 数据处理和加载
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

mnist_train = datasets.MNIST(root=r'E:\download', train=True, download=True, transform=trans)
mnist_val = datasets.MNIST(root=r'E:\download', train=False, download=False, transform=trans)

trainloader = DataLoader(mnist_train, batch_size=16, shuffle=True, num_workers=0)
valloader = DataLoader(mnist_val, batch_size=16, shuffle=True, num_workers=0)

# 模型
model = MLP()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 损失函数
celoss = nn.CrossEntropyLoss()
best_acc = 0


# 计算准确率
def accuracy(pred, target):
    pred_label = torch.argmax(pred, 1)
    correct = sum(pred_label == target).to(torch.float)
    # acc = correct / float(len(pred))
    return correct, len(pred)


acc = {'train': [], "val": []}
loss_all = {'train': [], "val": []}

for epoch in tqdm(range(40)):
    # 设置为验证模式
    model.eval()
    numer_val, denumer_val, loss_val = 0., 0., 0.
    with torch.no_grad():
        for data, target in valloader:
            output = model(data)
            loss = celoss(output, target)
            loss_val += loss.data

            num, denum = accuracy(output, target)
            numer_val += num
            denumer_val += denum

    # 设置为训练模式
    model.train()
    numer_tr, denumer_tr, loss_tr = 0., 0., 0.
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = celoss(output, target)
        loss_tr += loss.data
        loss.backward()
        optimizer.step()
        num, denum = accuracy(output, target)
        numer_tr += num
        denumer_tr += denum
    loss_all['train'].append(loss_tr / len(trainloader))
    loss_all['val'].append(loss_val / len(valloader))
    acc['train'].append(numer_tr / denumer_tr)
    acc['val'].append(numer_val / denumer_val)

plt.plot(loss_all['train'])
plt.plot(loss_all['val'])

plt.plot(acc['train'])
plt.plot(acc['val'])
