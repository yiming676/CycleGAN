import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import Generator
from datasets import ImageDataset
from torchvision.utils import save_image


"""
可以生成：
A：用橘子生成的苹果
B:用苹果生成的橘子
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

netG_A2B.load_state_dict(torch.load("models/netG_A2B.pth"))
netG_B2A.load_state_dict(torch.load("models/netG_B2A.pth"))

netG_A2B.eval()
netG_B2A.eval()
size = 256
input_A = torch.ones([1,3,size,size],dtype=torch.float).to(device)
input_B = torch.ones([1,3,size,size],dtype=torch.float).to(device)

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]

data_root = r"E:\算法学习\python_study\pytorch\4-CycleGAN\datasets\apple2orange"
dataloader = DataLoader(ImageDataset(data_root,transforms_,"test"),
                        batch_size = 1,shuffle=False,num_workers=0)

if not os.path.exists("outputs/A"):
    os.makedirs("outputs/A")
if not os.path.exists("outputs/B"):
    os.makedirs("outputs/B")
for i, batch in enumerate(dataloader):
    real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float).to(device)
    real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float).to(device)

    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    save_image(fake_A, "outputs/A/{}.png".format(i))
    save_image(fake_B, "outputs/B/{}.png".format(i))
    print(i)