import torch
import torch.nn as nn
from torchvision.models import resnet18
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
from tqdm import tqdm
import os

from dataset import PlantsDataset
from utils import parse_args

args = parse_args()             # 使用 argparse 套件方便微調程式

NUM_EPOCHS = args.epochs        # train the training data n times
BATCH_SIZE = args.batch_size
LR = args.lr                    # learning rate
NUM_WORKERS = args.num_workers
TRAIN_PATH = '../Fan-Kaggle-Plant-Seedlings-Classification-Example/train'
PRETRAINED_MODEL = args.pretrained_model_path

# 將資料讀到 DataLoader，並用自訂參數來調整 batch size 跟 num of workers
def load_plants_data():
    # 調整CPU數
    kwargs = {'num_workers': NUM_WORKERS}

    # data set 讀進來時要做的轉換
    data_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    # 將資料讀到 DataLoader，並用自訂參數來調整 batch size 跟 num of workers
    train_set = PlantsDataset(root_dir=Path(TRAIN_PATH), transform=data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                             shuffle=True, **kwargs)

    return train_set, data_loader

# Load resnet18 pretrained model
def load_model(num_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 18 層的 ResNet
    model = torch.hub.load('pytorch/vision:v0.5.0',
                           'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, num_class)  # by Fan (Not necessary)

    # Another method
    # model = resnet18(pretrained=True)

    # Load之前訓練model (Optional)
    if args.pretrained_model_path is not None:
        # model.load_state_dict(torch.load(PRETRAINED_MODEL))
        model = torch.load(PRETRAINED_MODEL)
    
    return model

def train():
    # 指定第幾張GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set, data_loader = load_plants_data()

    # 載入resnet18或之前訓練的model
    model = load_model(len(train_set))
    model.to(device)
    # model.cuda()        # by default will send your model to the "current device"

    # 定義 optimizer 跟 loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    # store model's best state_dict
    best_model_params = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch + 1}/{NUM_EPOCHS}')
        print('-' * len(f'Epoch: {epoch + 1}/{NUM_EPOCHS}'))

        training_loss = 0.0
        training_corrects = 0
        # data_loader內__getitem__定義回傳(inputs, labels)
        for i, (inputs, labels) in enumerate(tqdm(data_loader)):
            # Method 1
            # inputs = Variable(inputs.cuda())
            # labels = Variable(labels.cuda())

            # Method 2
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)         # Resnet18 output

            _, preds = torch.max(outputs.data, 1)   # 回傳 tensor 中的最大值，這邊為預測結果
            loss = criterion(outputs, labels)       # 計算 loss

            optimizer.zero_grad()           # 初始化梯度
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # 更新gradients參數(weight, bias) (optimizer內)

            training_loss += loss.data * inputs.size(0)
            training_corrects += (preds == labels.data).sum().item()  # 儲存正確的"數量"

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects / len(train_set)

        print(f'Training loss: {training_loss:.4f}\t accuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            # Delete previous modelpa
            if os.path.isfile(f'state_dict-{best_acc:.2f}-best_train_acc.pth'):
                os.remove(f'state_dict-{best_acc:.2f}-best_train_acc.pth')

            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

            # TODO:若save model成功就把state_dict刪掉
            # 每個Epoch存一次state_dict
            torch.save(model.state_dict(), f'state_dict-{best_acc:.2f}-best_train_acc.pth')

    model.load_state_dict(best_model_params)
    # 若單純test不能存state_dict
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')
    return model

if __name__ == '__main__':
    train()
