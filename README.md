# Kaggle-Plant-Seedlings-Classification
###### tags: `程式` `自學`

The objective is to train a Resnet18 model to classify plants from a picture.

## Getting Started

### Prerequsites
使用 Pytorch 1.4, Cuda 10.0

### Dataset
Download the dataset and put it in the root directory.
1. train
2. test
3. sample_submission.csv

[Plant Seedlings Classification | Kaggle](https://www.kaggle.com/c/plant-seedlings-classification/data)

### Training and Testing
**The predicted result will be stored as "submission.csv".**

```info=
# Execute
python main.py [--path] [--pretrained_model_path] [--epochs] [--batch_size] [--lr] [--num_workers] [--cuda]

[--path]: (str) image's path 
[--pretrained_model_path]: (str) pretrained model's path
[--epochs]: (int) epochs
[--batch_size]: (int) batch size
[--lr]: (float) learning rate
[--num_workers]: (int) number of cpu
[--cuda]: (Bool) use cuda or not

```
---

```python=
# EX. 
# "train" and "test" folder is in "./"
# pretrained model name is in "./final-0.93-best_train_acc.pth"
python main.py --path ./ --pretrained_model_path ./model-0.93-best_train_acc.pth 
```
![](https://i.imgur.com/y2POxRz.png)

## 整體架構
> Written by Fan

1. DataSet 讀取要用到的資料集
2. DataLoader 將當前要使用到的資料讀進來
3. 訓練模型(train)
4. 模型評估(test)

### 資料處理過程

1. 定義好 dataset 的 `__len__`, `__getitem__`, **transforms**
2. 資料進來到 dataset 後會根據 transforms 的格式作改變，將圖片跟label轉為 tensor
	- 意即將圖片轉為 1(batch size) + 3\*h\*w + 1(label) = 4 + 1維的 tensor
3. dataloader 從 dataset 中一次讀取 batch size 個資料進來

## train, eval

```python
model.train()
model.eval()
```

- 兩個功能都是告知模型現在要做甚麼
	- 在這邊沒什麼特別的用處
	- 如果使用像是 batch normalize 或 droup out 等，需要使用 train() 和 eval() 讓那些神經層能夠正確地操作

## Transforms

- 記得要注意 compose 有順序問題
	- 觀念跟 nn.sequential 差不多，都是 syntax candy 這類的東西
- transforms 有分對 Tensor 或對 PIL 檔案的函式，順序錯誤則無法使用

```python
transforms.Compose() 			# 作一系列的 transforms
transforms.RandomResizedCrop() 	# 隨機剪裁圖片
transforms.Scale()				# 設定圖片大小
transforms.ToTensor()			# 將資料轉為 tensor 的資料型態
transforms.Normalize()			# 正規化 tensor
```

- **224 是 ResNet 能輸入的最小圖片大小**

## ResNet

![](https://i.imgur.com/G3RjuFs.png)
- 定義好最後輸出的 fc層為多少就可以了，如果要改其他則自己根據需求更改

## Test

- unsqueeze 的用意為讓維度對齊
	- 在 dataset 中定義回傳為圖片+ batch size，所以維度為 3 + 1
	- 而 test set 中因為只對圖片作處理，並且一次使用全部圖片，所以維度為 3，故不符合模型需求的 input 維度
		- unsqueeze 為 tensor 物件中的函式

## 參考

### 官方文件

- [Training a classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Custom DataSet, DataLoader, and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [torch docs](https://pytorch.org/docs/stable/torch.html)
- [torchvision docs](https://pytorch.org/docs/stable/torchvision.html)

### 論文

- [ResNet 論文](https://arxiv.org/pdf/1512.03385.pdf)
