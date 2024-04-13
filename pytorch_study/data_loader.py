from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
test_data = torchvision.datasets.CIFAR10(r"C:\Users\lrd\Desktop\kaggle\pytorch\dataset",train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data1",imgs,step)
    step += 1
writer.close()
