# # 官方套件
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# from torchvision.models import resnet18

# 自訂
# from train import train
from train import train
from test import test

def main():
    model = train()
    test(model)

if __name__ == '__main__':
    main()
