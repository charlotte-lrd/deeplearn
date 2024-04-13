from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")
img_path = r"C:\Users\lrd\Desktop\kaggle\pytorch\data\train\bees_image\16838648_415acd9e3f.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
# image可以是tensor opencv
writer.add_image("test",img_array,2,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=4x",2*i,i)
writer.close()