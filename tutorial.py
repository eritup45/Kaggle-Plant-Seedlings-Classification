import torch.nn as nn
# from torchvision.models import resnet18

"""             各種模型輸出方式            """

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.add_module("conv", nn.Conv2d(10, 20, 4))
        self.add_module("conv1", nn.Conv2d(20 ,10, 4))

model = Model()

# 返回一个包含当前模型所有模块的迭代器
# for module in model.modules():
#     print(module)
# print('-----')

# 返回当前模型子模块的迭代器
# for sub_module in model.children():
#     print(sub_module)

# 返回包含模型当前子模块的迭代器
# for name, module in model.named_children():
#     if name in ['conv1', 'conv5']:
#         print(module)

# 返回一个 包含模型所有参数 的迭代器。
# 一般用来当作optimizer的参数。
# for param in model.parameters():
#     print(type(param.data), param.size())
# print('-----')

# print(f'{model.state_dict()}')
print(model)

"""         參考網址：https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305d9cd231015d9d0992ef0030      """
"""         常使用的tensor操作          """

"""         創造矩陣                    """
torch.ones(5, 3)    # 創造一個填滿1的矩陣
torch.zeros(5, 3)   # 創造一個填滿0的矩陣

torch.eye(4)        # 創造一個4x4的單位矩陣

torch.rand(5, 3)    # 創造一個元素在[0,1)中隨機分佈的矩陣
torch.randn(5, 3)   # 創造一個元素從常態分佈(0, 1)隨機取值的矩陣

"""         矩陣操作                    """
torch.cat((m1, m2), 1)    # 將m1和m2兩個矩陣在第一個維度合併起來
torch.stack((m1, m2), 1)  # 將m1和m2兩個矩陣在新的維度（第一維）疊起來

m.squeeze(1)              # 如果m的第一維的長度是1，則合併這個維度，即
                          # (A, 1, B) -> (A, B)
m.unsqueeze(1)            # m的第一維多一個維度，即
                          # (A, B) -> (A, 1, B)
 
m1 + m2                   # 矩陣element-wise相加，其他基本運算是一樣的

"""         其他重要操作                """
m.view(5, 3, -1)    # 如果m的元素個數是15的倍數，回傳一個大小為(5, 3, ?)的
                    # tensor，問號會自動推算。tensor的資料是連動的。
m.expand(5, 3)      # 將m擴展到(5, 3)的大小

m.cuda()            # 將m搬移到GPU來運算
m.cpu()             # 將m搬移到CPU來運算

torch.from_numpy(n) # 回傳一個tensor，其資料和numpy變數是連動的
m.numpy()           # 回傳一個numpy變數，其資料和tensor是連動的

"""         Variable                    """
"""         Variable的操作除了data的資料會有改動，所有的操作也會記錄下來變成一個有向圖  """
import torch
from torch.autograd import Variable
from torch.optim import SGD

m1 = torch.ones(5, 3)
m2 = torch.ones(5, 3)

# 記得要將requires_grad設成True
a = Variable(m1, requires_grad=True)
b = Variable(m2, requires_grad=True)

# 初始化優化器，使用SGD這個更新方式來更新a和b
optimizer = SGD([a, b], lr=0.1)

for _ in range(10):        # 我們示範更新10次
    loss = (a + b).sum()   # 假設a + b就是我們的loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()       # 更新

"""         Module(模組)                """
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        """
        在__init__函數裡定義這個模組會用到的參數
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       """
       在forward函數裡定義輸入和輸出值的關係
       """
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
       
# 假設 _input是一個變數
model = Model()
y = model(_input)    # y就是我們模組的輸出           






