from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import torch
import torchvision
# 使用transform批量处理数据
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)
