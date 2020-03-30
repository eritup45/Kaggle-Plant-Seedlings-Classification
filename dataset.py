from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class PlantsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.img = [] # img
        self.idx = [] # index
        self.transform = transform

        if self.root_dir.name == 'train':
            for i, _dir in enumerate(self.root_dir.glob('*')):
                for img in _dir.glob('*'):
                    self.img.append(img)
                    self.idx.append(i)
        else:
            print('WARNING:Can not find folder \"train\" !')

    # =============================================================== #
    # 需要自己 override 的兩個函式 
    # =============================================================== #

    # 回傳 data set 的長度，以避免 data loader 抓取資料的時候超出範圍
    def __len__(self):
        return len(self.img)

    # 定義 data loader 拿到的資料長甚麼樣子
    def __getitem__(self, index):
        image = Image.open(self.img[index]).convert('RGB')

        if self.transform:
            # 官方建議寫法，確保每個進來的資料大小要一樣
            image = self.transform(image)

        # 回傳兩個 tensor，片段圖片資訊跟對應的label
        return image, self.idx[index]