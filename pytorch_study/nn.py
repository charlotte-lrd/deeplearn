from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data = torchvision.datasets.CIFAR10(r"C:\Users\lrd\Desktop\kaggle\pytorch\dataset",train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,padding=0)
        # 可以用数组提供
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=4,ceil_mode=False)
        self.batchnorm = nn.BatchNorm2d(3)
        self.linear = nn.Linear(in_features=6*7*7,out_features=3*7*7)
        #输入数据需要展平为二维，第一维为batchsize 第二维度为in_features
        self.dropout = nn.Dropout2d(p=0.1)
        self.model = nn.Sequential(self.batchnorm,self.conv1,self.relu,self.maxpool)

    def forward(self,x):
        # x = self.batchnorm(x)
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x = self.model(x)
        x = torch.reshape(x,(x.shape[0],-1))
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
model = Test()
step = 0
writer = SummaryWriter("nn")
for data in test_loader:
    imgs, targets = data
    output = model(imgs)
    output = torch.reshape(output,(-1,3,7,7))
    writer.add_images("nn_dropout_output",output, step)
    step += 1
writer.close()