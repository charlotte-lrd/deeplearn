from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
np.ndarray
# 使用transform批量处理数据
#设置transform处理器
dataset_transforms = transforms.Compose([
    transforms.ToTensor()
])
#批量处理数据
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transforms)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transforms)
#显示在tensorboard上
writer = SummaryWriter('logs_batch')
for i in range(0,10):
    img , target = test_set[i]
    writer.add_image('test_set',img,i)
writer.close()