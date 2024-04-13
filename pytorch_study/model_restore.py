import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
vgg16 = torchvision.models.vgg16(pretrained=False)
#想把它修改成训练10分类的模型
# vgg16.add_module("add_linear",nn.Linear(1000,10))
# print(vgg16)

#另一个方式
vgg16.classifier[6] = nn.Linear(4096, 10)


test_data = torchvision.datasets.CIFAR10(r"C:\Users\lrd\Desktop\kaggle\pytorch\dataset",train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
#保存模型
torch.save(vgg16, "vgg16_method1.pth")
#加载模型
model = torch.load("vgg16_method1.pth")
print(model)