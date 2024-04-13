from torch import nn
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
train_data = torchvision.datasets.CIFAR10(r"C:\Users\lrd\Desktop\kaggle\pytorch\dataset",train=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = torchvision.datasets.CIFAR10(r"C:\Users\lrd\Desktop\kaggle\pytorch\dataset",train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64,out_features=10)

        self.model1 = nn.Sequential(
        self.conv1,
        self.maxpooling,
        self.conv2,
        self.maxpooling,
        self.conv3,
        self.maxpooling,
        self.flatten,
        self.linear1,
        self.linear2
        )

    def forward(self,x):
        x = self.model1(x)
        return x
writer = SummaryWriter(r"./pytorch/cifar10")
model = CIFAR10()
step = 0
step_test = 0
loss_ce = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=1e-3)
num_epochs = 5
if_graph = 1
#SGD:随机梯度下降 lr:学习率 momentum:L2范数前系数 dampening:动量法阻尼 nesterov:某种动量法
for epoch in range(num_epochs):
    loss_epoch = torch.tensor([0.])
    for data in train_loader:
        imgs, targets = data
        # writer.add_graph(model, imgs)
        # # 只能记录一个图
        output = model(imgs)
        loss = loss_ce(output, targets)
        loss_epoch += loss
        optimizer.zero_grad() #用于清空param的grad
        loss.backward() #计算出param此循环中的grad,将grad加在param的grad中
        optimizer.step() #根据param的grad和optimizer的准则，更新param的值
        writer.add_scalar('loss/step:',loss.item(),step)
        step += 1
        if if_graph:
            if_graph = 0
            writer.add_graph(model,imgs)
    writer.add_scalar('loss/epoch:',loss_epoch.item(),epoch)
    print("Epoch:{}/10 loss:{}".format(epoch+1,loss_epoch.item()))

    loss_test = torch.tensor([0.])
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_ce(output, targets)
            loss_test += loss
            writer.add_scalar('test_loss/step:',loss.item(),step_test)
            step_test += 1
    writer.add_scalar('test_loss/epoch:',loss_test.item(),epoch)
writer.close()



