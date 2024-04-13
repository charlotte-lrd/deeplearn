from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import torch

img_path = r"C:\Users\lrd\Desktop\kaggle\pytorch\data\train\ants_image\0013035.jpg"
img = Image.open(img_path)

# Compose:批量化处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为 224x224
    transforms.PILToTensor()        # 将图像转换为张量
])
transformed_img = transform(img)

# ToTensor:numpy&PIL 转化为tensor张量
writer = SummaryWriter("logs_tf")
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor (img)
##writer.add_image("ToTensor", img_tensor, 1)

#topilimage:tensor数据类型转化为PIL数据类型用于输出(可视化)
tensor_img = torch.rand(3, 100, 100)
# 创建 ToPILImage 变换操作
to_pil_image = transforms.ToPILImage()
# 将张量形式的图像转换为 PIL 图像
pil_img = to_pil_image(tensor_img)

#normalize:tensor在每个channel上归一化（但是要自己输入mean,std)

# 假设我们有一个均值和标准差
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# 创建 Normalize 变换操作
normalize = transforms.Normalize(mean, std)
writer.close()
