## 1.反向传播

主体流程：

1.先计算loss函数

2.对计算出来的loss值进行反向传播

3.更新梯度

注：

1.w.grad也是一个Tensor 进行数值计算时需要对其取.data,若直接使用w.grad进行计算，则会生成一个计算图

2.计算出来的loss（也是一个tensor）值不能直接进行加法运算，需要将其中的值取出来再进行计算用到了item()函数，形式为loss.item()

3.每次进行一轮梯度迭代时需要将梯度清零，否则会将每一轮的梯度进行叠加，w.grad.data.zero_()

```python
import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]
#初始化权值w
w = torch.Tensor([1.0])
w.requires_grad = True    #计算梯度设置为true
lr = 0.01 #学习率

def forward(x):
    """
    前馈函数
    """
    return x * w #因为w是tensor,这里在计算时默认将x也转化为tensor

def loss(x, y):
    """
    计算损失值（也是优化的目标函数）
    """
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict(before training)", 4, forward(4).item())  #??
for epoch in range(100):  #每一次迭代
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  #计算损失函数
        l.backward()  #反向传播
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - lr * w.grad.data  #跟新梯度
        w.grad.data.zero_()  #将每一轮的梯度清零
    print("progess:", epoch, l.item())
print(w)
print("predict(before training)", 4, forward(4).item())

```

## 2.线性回归

1.准备数据集

2.设计模型 

3.构造损失函数和优化器（利用pytorch API）

4.训练周期（前馈算损失、反馈算梯度、更新权重）

若有三个样本（x1, y1）（x2, y2）（x3, y3）

![image-20220107210417548](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220107210417548.png)

w和b都会进行广播

![image-20220107210505330](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220107210505330.png)

则，损失的表达形式为：

![image-20220107210759610](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220107210759610.png)

#### 构造数据

在构造初始数据时，x和y必须是(3, 1)的矩阵的形式（也可看作列向量）

```python
import torch
#注意x和y都是列向量，一个中括号代表一行，放入一个数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
```

#### pytorch中的分析

使用pytorch时，我们的重点不在求导，而在于构造计算图

![image-20220107211324327](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220107211324327.png)

其中z = w * x + b为仿射函数，又叫做线性单元

**需要确定x和y的维度，即可确定w的维度。将计算得到的y放入loss函数，并对得到的loss进行反向传播**

若计算得到的loss为一个向量，需要将其转化为标量，做法为：将所有loss求和取平均值(也可以只加和不求均值)。

#### 构造模型

将我们的模型构造为一个类(都要继承自module)：

```python
class LinearModel(torch.nn.Module): #从Module继承
    #必须实现以下两个函数
    #初始化
    def __init__(self):
        super(LinearModel, self).__init__()  #调用父类的初始化
        self.linear = torch.nn.Linear(1, 1)  #构造一个对象，包含权重和偏置
        #Linear的参数为，输入的维度（特征数量，不是样本数量）和输出的维度，以及是否有偏置(默认为True)
    #前馈过程中进行的计算
    def forward(self, x):  #这里实际上是一个override
        y_pred = self.linear(x)  #在这里计算w * x + b 线性模型
        return y_pred
    #不需要自己写反馈

model = LinearModel()   #这里的model是callable可调用的 调用方法为:model(x)
```

#### 构造损失函数和优化器

```python
criterion = torch.nn.MSELoss(size_average=False) #是否要求均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#优化器不会构建计算图， 在这里需要告诉优化器对那些Tensor进行梯度下降
#model.parameters()函数会检查模型中的成员变量，将成员变量里相应的权重加入训练的结果中
```

SGD的参数如下：

![image-20220107214352550](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220107214352550.png)

#### 迭代更新梯度

前馈算y'、损失、梯度归零、反向传播、自动更新

```python
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()   #梯度归零
    loss.backward()
    optimizer.step()  #根据梯度和预先设置的学习率自动更新
```

#### 输出与测试

```python
#输出权重和偏置
print('w = ', model.linear.weight.item())
#weight是一个矩阵，需要显示数值用item()
print('b = ', model.linear.bias.item())

#测试
x_test =torch.Tensor([[4.0]])   #(1, 1)矩阵
y_test = model(x_test)
print('y_pred = ', y_test.data) #(1, 1)矩阵
```

## 3.逻辑回归（分类）

核心任务：计算属于每一类的概率：

数据集的下载：

```python
train_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='../dataset/mnist', train=False, download=True)
```

逻辑回归损失用交叉熵BCE（二元）：

![image-20220109184638154](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220109184638154.png)

将小批量数据的所有BCE损失求和并求均值：

![image-20220109184831864](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220109184831864.png)

#### 代码实现

```python
import torch.nn
import torch.nn.functional as F

#准备数据
x_data = torch.Tensor([[1], [2], [3]])
y_data = torch.Tensor([[0], [0], [1]])

#构造模型
class LogisticRegressionModel(torch.nn.Module):
    #初始化与线性模型一样，因为σ没有需要训练的参数
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

#构造损失和优化器
#这里采用交叉熵
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#训练的循环
for epoch in range(1000):
    y_pred = model(x_data)  #前馈
    l = criterion(y_pred, y_data)  #计算损失
    print(epoch, l.item())

    optimizer.zero_grad()   #梯度清零
    l.backward()  #反向传播
    optimizer.step()   #更新梯度

#预测
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1, 10, 200)
x_test = torch.Tensor(x).view((200, 1))  #这里相当于一个reshape
y_test = model(x_test)
y = y_test.data.numpy()  #在这里需要将Tensor转化回numpy
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()
```

## 4.处理多维特征的输入

以8维特征为例，模型表达式如下：

![image-20220110112333866](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110112333866.png)

则，N个样本8个特征的线性表达式为：

![image-20220110115636240](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110115636240.png)

构建模型时，仅需要将输入的维度进行改动即可（输入为8，输出为1）：

![image-20220110120836894](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110120836894.png)

![image-20220110121039942](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110121039942.png)

采用多层神经网络结构：

![image-20220110141907460](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110141907460.png)

#### 准备数据

从diabetes.csv中读取数据：输入（759， 8）、输出（759， 1）

```python
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])  #若这里的-1不加括号，生成的数据为一维
```

#### 模型构造

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  #将sigmoid作为一个运算模块（没有参数），不在前馈中出现，放入初始化中

    def forward(self, x):
       x = self.sigmoid(self.linear1(x))
       x = self.sigmoid(self.linear2(x))
       x = self.sigmoid(self.linear3(x))
       return x
model = Model()
```

#### 构造损失函数和优化函数

```python
#构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=True)
#SGD是随机梯度下降法
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

#### 训练

```python
for epoch in range(100):
    #前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)
    #反向传播
    optimizer.zero_grad()  #先清零
    loss.backward()
    #自动更新梯度
    optimizer.step()
```

## 5.加载数据集

1.每次采用一个样本进行随机梯度下降，会得到随机性较好的训练结果，但是速度较慢，训练时间长

2.加入batch，用全部的样本进行训练时速度十分快，但是会训练效果会下降。

所以在这里引入**mini-batch**，综合二者的优点，得到折衷的结果。

batch_size（批量大小）：进行一次前馈、反馈、更新所用到的样本数量

iteration(迭代次数)：样本数量/批量大小

#### 具体实现

1.把所有的数据加载进来，在使用getitem时将第i个样本传出去（适合样本总量较小的数据集）

2.定义一个列表，将文件名放入列表中，在使用getitem时，读取第i个文件中的内容（如几十个G的图像集）

```python
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):   #由Dataset(抽象类)继承
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):  #通过索引得到数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv')  #实例化为一个对象
#dataset每次可以拿到一个数据样本
#加载器  DataLoader根据batch_size的数量将多个dataset的x和y组合成矩阵，并自动生成Tensor
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2   #多线程
                          )
```

相应的训练过程修改为：

```python
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        #得到数据
        inputs, labels = data
        #前馈
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        #更新
        optimizer.step()
```



注：

使用多线程时，需要将训练过程封装，因为linux中使用fork创建多线程，windows中使用spawn创建多线程

![image-20220110172022054](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110172022054.png)

读取MNINST:

![image-20220110174108922](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110174108922.png)

练习：

![image-20220110174305038](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110174305038.png)

## 6.多分类问题

#### softmax

**核心：最后一层使用softmax层**

![image-20220110175026586](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110175026586.png)

1.求指数将负值转化为非负值

2.分母将所有输出求和（归一化）

保证条件如下：

![image-20220110175038774](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110175038774.png)

#### 损失函数

使用负对数似然函数（只有y=1的项才真正被计算，为0不影响结果）：

![image-20220110175647921](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110175647921.png)

交叉熵损失函数的使用

1.不需要单独使用sofmax（已经被包含在CrossEntropyLoss()函数中）

2.y需要为长整型张量，使用LongTensor()进行构造

```python
import torch

y = torch.LongTensor([0])
z = torch.Tensor([[0.2, 0.1, -0.1]])
criterion = torch.nn.CrossEntropyLoss()
#交叉熵损失也包括了softmax，所以不需要单独似乎用激活函数
loss = criterion(z, y)
print(loss)
```

交叉熵损失和NLL损失的区别：

![image-20220110201930031](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110201930031.png)

#### 多分类的实现

##### 数据的准备

注意：

1.神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
2.**通道**：仅有灰度值->单通道   RGB -> 三通道 

3.读入的图像张量一般为W * H* C (宽、高、通道数) 在pytorch中要转化为C * W * H

在本例中需要将输入的图像数据（如下图所示）:

**a由(28，28)转化为（1， 28， 28）**

**b数值范围从{0，...，255}转化为{0，...，1}**

![image-20220110220939573](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220110220939573.png)

这一步可以由transfroms中的ToTensor()完成

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.optim as optim

#数据的准备
batch_size = 64
#神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
#仅有灰度值->单通道   RGB -> 三通道 读入的图像张量一般为W*H*C (宽、高、通道数) 在pytorch中要转化为C*W*H
transform = transforms.Compose([
    #将数据转化为图像张量
    transforms.ToTensor(),
    #进行归一化处理，切换到0-1分布 （均值， 标准差）
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                          )
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform
                               )
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size
                          )

```

##### 构建模型、损失函数及优化器

注：在ToTensor处理完后，选取batch_size为一组后，数据为（N， 1， 28，28）我们需要将其转化为可以输入的矩阵形式，将（1， 28， 28）变为一维的向量(784, )  生成的输入矩阵为（N， 784）

```python
#构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) #将数据转化为二维矩阵，可输入神经网络
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)    #最后一层不需要激活函数，后面会接入softmax

model = Net()

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

#### 训练及测试部分

```python
#训练
#将一次迭代封装入函数中
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))


def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        for data in test_loader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            outputs = model(images)#得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)#dim=1沿着索引为1的维度(行)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```



#### 训练结果

如下图所示，从第一个batch开始，没个batch内输出三次损失值，和一次准确率

![image-20220111133057365](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220111133057365.png)

经过十次迭代后，准确率上升到了97%，损失值也几近收敛

![image-20220111133249372](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220111133249372.png)

## 7.卷积神经网络（基础）

#### 前言

若将数据输入全连接层，可能会导致丧失一些位置信息

卷积神经网络将图像按照原有的空间结构保存，不会丧失位置信息。

1.栅格图像

2.矢量图像（人工生成）

做完卷积后，通道数、宽和高都有可能改变

#### 卷积运算：

1.以单通道为例：

将将input中选中的部分与kernel进行数乘 ：

![image-20220112155432573](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112155432573.png)

以上图为例对应元素相乘结果为211，并将结果填入output矩阵的左上角

![image-20220112155553493](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112155553493.png)

得到：

![image-20220112155709863](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112155709863.png)

最终得到的结果为：

![image-20220112155731931](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112155731931.png)

2.三通道卷积

![image-20220112155945445](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112155945445.png)

三个通道分别利用三个卷积核进行计算，并将结果相加得到最终定格结果。

那么我们可以得到n个通道的卷积过程

![image-20220112160134739](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112160134739.png)

若希望得到的卷积后结果通道数为m，则需要m个卷积核同时对原始数据进行处理，得到m个结果，并将其拼接起来，也得到了m个通道

![image-20220112160320166](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112160320166.png)

注意：

1.每一个卷积核的通道数量和输入的通道数量是相同的

2.卷积核的总数与输出的通道数量相等

所以我们要构建一个卷积层，其权重的维度为：

![image-20220112160837259](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112160837259.png)

**代码演示：**

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

![image-20220113115950846](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113115950846.png)

```python
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3  #卷积核的大小
batch_size = 1

input = torch.randn(  #正态分布随机采样
    batch_size,
    in_channels,#这个参数一定要前后对应，否则无法正常运行，其余参数无所谓
    width,
    height
)

#生成一个卷积对象   至少需要三个参数
conv_layer = torch.nn.Conv2d(in_channels,  #输入通道的数量
                             out_channels, #输出通道的数量
                             kernel_size=kernel_size) #卷积核尺寸

output = conv_layer(input)

print(input.shape)
#torch.Size([1, 5, 100, 100])
print(output.shape)
#torch.Size([1, 10, 98, 98])
#相比于100减去了2，是因为采用了3*3的卷积核，将宽度为2边框去除了
print(conv_layer.weight.shape)
#torch.Size([10, 5, 3, 3])
#10为m输出的通道数  5为n输入的通道数  3*3为卷积核的尺寸
```

#### 卷积运算中几个常用的参数

##### 1.padding

不想改变输出矩阵的大小，可以向外填充，默认是填充零

![image-20220112162916859](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112162916859.png)

**代码演示：**

```python
import torch

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)  #四个参数分别为B C W H
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

#构造卷积核
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3) #参数为 输出通道数, 输入通道数, W, H
conv_layer.weight.data = kernel.data  #将构造的卷积核赋值给卷积层的权重  记得要加.data

output = conv_layer(input)
print(output)
#卷积计算的结果如下
#tensor([[[[ 91., 168., 224., 215., 127.],
#          [114., 211., 295., 262., 149.],
#          [192., 259., 282., 214., 122.],
#          [194., 251., 253., 169.,  86.],
#          [ 96., 112., 110.,  68.,  31.]]]], grad_fn=<ThnnConv2DBackward0>)
```

##### 2.stride

W*H框运动的步长，可以有效降低图像的宽度和高度

例如，当stride=2时，通过3*3卷积核计算得到的输出维度为2 * 2

![image-20220112164233710](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112164233710.png)

**代码演示：**

```python
#在上一小节的代码中做如下修改即可
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
```

##### 3.Max Pooling Layer 

最大池化层：将图像分成2*2的小块，在每一块中寻找最大值，返回给输出，如下图所示，将4 * 4的图像经过池化后变为了2 * 2的图像

作用：减少数据量，降低运算需求

注意：池化层操作只针对同一通道，对通道的数量等其他参数无影响



![image-20220112164601576](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112164601576.png)

**代码演示：**

```python
import torch

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6]
input = torch.Tensor(input).view(1, 1, 4, 4)  #输出输入通道都为1，卷积核尺寸为4*4

maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)  #相当于stride=2,取2*2为一组

output = maxpooling_layer(input)
print(output)
#tensor([[[[4., 8.],
#          [9., 8.]]]])
```

#### 实战演练

数据采用MNIST

##### 设计一个卷积神经网络

![image-20220112170009862](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112170009862.png)

如上图所示，将一个（1， 28， 28）的图像张量输入进卷积网络中

1.首先经过一个卷积核大小为(5， 5) 输出通道数为10的卷积层

2.经过一个(2， 2)的最大池化层（取出2*2范围内的最大值返回给输出）

3.再经过一个卷积核大小为(5， 5) 输出通道数为20的卷积层，在这里通道数由10变为20

4.又经过一个(2， 2)的最大池化层

5.最终通过一个输入为320维输出为10为的全连接层（作为分类器），得到10个概率值对应于10种类型  320由上一层的参数总数计算而来（20 * 4 * 4）

注：运算过程中，池化层不关注输入输出的维度，但是卷积层和全连接层的输入和输出维度一定要和前后层相对应。

由此，我们得到每一步的核心函数：

![image-20220112170837075](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112170837075.png)

**代码实现：**

```python
import torch
import torch.nn.functional as F #使用functional中的ReLu激活函数

#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        #池化层
        self.pooling = torch.nn.MaxPool2d(2)  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
model = Net()
```

##### GPU的使用

1.将模型迁移到GPU

```python
#在model = Net()后加入如下代码
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中
```

2.训练中将数据迁入GPU，包括inputs和target

```python
#inputs, target = data后加入.to(device)
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        #在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
```

3.测试中同样也需要将数据送入GPU中，代码同上

##### 整体代码：

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.optim as optim

#数据的准备
batch_size = 64
#神经网络希望输入的数值较小，最好在0-1之间，所以需要先将原始图像(0-255的灰度值)转化为图像张量（值为0-1）
#仅有灰度值->单通道   RGB -> 三通道 读入的图像张量一般为W*H*C (宽、高、通道数) 在pytorch中要转化为C*W*H
transform = transforms.Compose([
    #将数据转化为图像张量
    transforms.ToTensor(),
    #进行归一化处理，切换到0-1分布 （均值， 标准差）
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform
                               )
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size
                          )
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=False,
                               download=True,
                               transform=transform
                               )
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size
                          )


#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        #池化层
        self.pooling = torch.nn.MaxPool2d(2)  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        return x
model = Net()
#在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中

#设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
#神经网络已经逐渐变大，需要设置冲量momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#训练
#将一次迭代封装入函数中
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):   #在这里data返回输入:inputs、输出target
        inputs, target = data
        #在这里加入一行代码，将数据送入GPU中计算！！！
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        #前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

def test():
    correct = 0
    total = 0
    with torch.no_grad():  #不需要计算梯度
        for data in test_loader:   #遍历数据集中的每一个batch
            images, labels = data  #保存测试的输入和输出
            #在这里加入一行代码将数据送入GPU
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)#得到预测输出
            _, predicted = torch.max(outputs.data, dim=1)#dim=1沿着索引为1的维度(行)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

##### 运行结果

我只进行了10轮迭代，准确率最终达到了98%,与单纯使用多层的全连接神经网络准确率97%相比提高了一个百分点，错误率降低了1/3，性能有了显著地提升

另外本次在GPU上运行十轮迭代的时间约为1-2分钟，请大家放心跑，无压力。

![image-20220112174139420](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220112174139420.png)

## 8.卷积神经网络（高级）

#### 分析

在上一节中使用的是串联的网络结构

 在本节中采用更为复杂的网络结构

以GoogLeNet为例：

![image-20220113111917279](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113111917279.png)

该网络结构十分复杂，由串联和并联多部分结构组成，但是我们可以观察到红框中的Inception Module有多次的重复，可以通过复用来降低代码量

其中的1个Inception如下所示：

![image-20220113111852953](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113111852953.png)

我们看到这个网络结构中出现了1 * 1的卷积核，那么为什么要使用1 * 1的卷积核呢？

![image-20220113112453115](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113112453115.png)

如上图所示，如果仅采用5 * 5的卷积核进行运算，那么我们的计算量是in_channels * Win * Hin * kernel_size ^ 2 * out_channels * Wout * Hout = 120，422，400

若先经过1 * 1的卷积运算将通道数将为16后在进行5 * 5的卷积运算，那么运算量会减少到十分之一

由此我们可以得到，通过1 * 1的卷积运算可以大大降低卷积运算的运算量

#### 模型实现

拿出上述Inception单独建立一个类，其中实现每一层的核心函数如下所示：

![image-20220113121138205](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113121138205.png)

最终需要将得到的四个输出沿通道拼接在一起，卷积网络中张量的维度为（B, C, W, H） 则参数dim=1就为沿着channel维度进行拼接

![image-20220113121215083](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220113121215083.png)

**inception实现**

```python
import torch
import torch.functional as F

class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):   #将输入通道数作为参数输入
        super(InceptionA, self).__init__()
        #分支1  输出通道数为24的1 * 1卷积层
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)
        #分支2  输出16通道的1*1卷积层
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        #分支3  16输出的1*1卷积 + 24输出的5*5卷积
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)  #若想保持图像张量大小不变padding=[kernel_size/2]
        #分支4 16输出的1*1卷积 + 24输出的3*3卷积 + 24输出的3*3卷积
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        """
        前向运算过程，共有四个分支，每个分支独立计算，并将四个计算结果沿通道方向进行拼接
        :param x:  每个分支的输入都为x
        :return:   返回四个分支输出的拼接结果
        """
        #分支1  均值池化+1*1卷积 均值池化利用functional中的函数完成,卷积利用定义好的成员变量完成
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        #分支2
        branch1x1 = self.branch1x1(x)
        #分支3
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        #分支4
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        #将4个分支得到的结果进行拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1) #在1维度将三个图像张量进行拼接
```

**基于Inception的模型实现**

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        #输入通道数为88是将inception中四个分支拼起来的结果24*3+16=88
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forwad(self, x):
        in_size = x.size(0)  #得到第一维的size,即batch_size
        x = F.relu(self.mp(self.conv1))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2()
        x = x.view(in_size, -1)  #每个batch为一行
        x = self.fc(x)
        return x
```

其余步骤与卷积神经网络基础中一致

## 9.Residual Net









## 10.循环神经网络

#### RNN基础

![image-20220115101442834](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115101442834.png)

在单个RNN神经元，mw维输入经过线性变换得到n维的输出

在多个神经元中，利用上一层的输出ht-1和本层的状态xt经过神经元的线性变换得到输出ht

RNN着重于解决序列问题，引入状态变量，每一个状态的产生依赖于前面的若干状态

![image-20220115105231736](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115105231736.png)

如上图所示，绿框即为前后状态组合计算的方法。

##### 创建一个RNNcell:

```python
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)  #需要输入的参数为：输入维度大小和隐层维度
```

简单的实践：

对一个长度为三的序列构建RNNcell模型

```python
import torch

#参数设置
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

#数据集维度 (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('input size: ', input.shape)

    hidden = cell(input, hidden)

    print('output size: ', hidden.shape)
    print(hidden)
```

##### 构建一个RNN模型

![image-20220115112915735](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115112915735.png)

```python
cell = torch.mm.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
out, hidden = cell(inputs, hidden)  
#inputs为整体的输入序列x1....xn     out为h1...hn, hidden为hn
#输入维度要求 (seqsize, batch, input_size)
#隐层维度要求 (numlayers, batch, hidden_size)
```

可能存在多层RNN，此时numlayers>1

![image-20220115113249656](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115113249656.png)

##### 实战

任务：利用RNN建模，完成序列到序列的转化（hello -> ohlol）， 学习序列变换的规律：

![image-20220115115301625](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115115301625.png)

独热向量转化方式如下：

![image-20220115115340608](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115115340608.png)

模型结构:

![image-20220115115442751](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220115115442751.png)

采用RNNCell实现：

```python
import torch

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']  #字典
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]   #得到的向量为(seq, input_size)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)   #分类标签不能是float，必须是long

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):  #batch_size只有在构造h0时需要
        return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

#训练
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()  #先进性梯度清零
    hidden = net.init_hidden()
    print('Predicted string: ', end='')
    #inputs的维度为(seqLen, batch_size, input_size)  input维度为(batch_size, input_size)
    for input, label in zip(inputs, labels):
        hidden = net.forward(input, hidden)
        loss += criterion(hidden, label)  #计算每一项损失的和，需要构造计算图，不能用.item()
        _, idx = hidden.max(dim=1)  #因为采用RNNCell中hidden的维度为(batch_size, hidden_size)
        print(idx2char[idx.item()], end=' ')
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
```

采用RNN实现：

```python
import torch
from icecream import ic

input_size = 4
hidden_size = 4
batch_size = 1
num_layers = 1

idx2char = ['e', 'h', 'l', 'o']  #字典
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]   #得到的向量为(seq, input_size)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)   #分类标签不能是float，必须是long

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=num_layers):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        #初始化h0
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        #reshape to (seqLen*batch_size, hidden_size)
        return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

#训练
for epoch in range(15):
    optimizer.zero_grad()  #先进性梯度清零
    outputs = net(inputs)
    loss = criterion(outputs, labels)  #outputs (5, 4)  labels (5, )
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join(idx2char[x] for x in idx), end='')
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
```

注：在这里使用的torch.nn.CrossEntropyLoss()有两个参数维度分别为（m, n）和(m)  其中m为样本个数，n为分类个数  第一个参数每一行的的内容为softmax的输入，对应每个分类结果的权重，第二个参数则为每个样本的实际分类。

![image-20220116123029350](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220116123029350.png)



##### 使用embedding和linear layer构建循环神经网络

加入embedding和linear layer后的网络结构

![image-20220116122942316](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220116122942316.png)







##### LSTM

