from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        '''
        get data and its label
        '''
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        '''
        get the length of data
        '''
        return len(self.img_path)
    
root_dir = r"C:\Users\lrd\Desktop\kaggle\pytorch\data\数据集\hymenoptera_data\hymenoptera_data\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_data_set = MyData(root_dir, ants_label_dir)
bees_data_set = MyData(root_dir, bees_label_dir)
train_dataset = ants_data_set + bees_data_set
