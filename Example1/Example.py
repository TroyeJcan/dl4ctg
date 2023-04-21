# -*- coding: utf-8 -*-
"""
Created on 2023/4/3 11:30
Location CellsVision

@author: Troye Jcan
"""
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 自定义数据集，继承torch.utils.data.Dataset，一般的数据集只需要重写下面的三个方法即可
class MyDataset(Dataset):
    def __init__(self, data_len, ):
        """ 初始化数据集并进行必要的预处理
        初始化数据集一般需要包括数据和标签：数据可以是直接可以使用的特征或路径；标签一般可以存放在csv中，可以选择读取为列表
        """
        # 随机初始化一个shape为(data_len, 10)的矩阵作为输入数据x，数据类型为float
        self.datas = torch.randn(size=(data_len, 10), dtype=torch.float32)
        # 随机初始化一个shape为(data_len, 1)的矩阵作为标签y，数据类型为int
        self.labels = torch.randint(low=0, high=10, size=(data_len, 1))

    def __len__(self):
        """ 返回数据集的大小，方便后续遍历取数据 """
        return len(self.labels)

    def __getitem__(self, item):
        """ item不需要我们手动传入，后续使用dataloader时会自动预取
        这个函数的作用是根据item从数据集中取出对应的数据和标签
        """
        return self.datas[item], self.labels[item][0]


# 定义一个含有100条数据的数据集
dataset = MyDataset(100)
# 使用dataloader每次从dataset中取出batch_size数目的数据，shuffle表示随机打乱数据集
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 可以打印看看数据集的shape是什么样的，可以看到dataset的长度为 data_len，而loader的长度为 data_len // batch_size
print(len(dataset))  # shape = data_len
print(len(loader))  # shape = data_len // batch_size
# output:
# 100
# 50

# 在dataloader中取数据时可以使用enumerate枚举的方法，因为枚举时会输出当前的数据是第几批，可以作为第几个batch的标识
# 很多人在训练时都会在第 batch % 50 == 0 为条件，输出每50个batch的模型表现
for batch, (data, label) in enumerate(loader):
    print(data)
    print(label)
    break

# 当然也可以直接循环取数据，效果都一样
for data, label in loader:
    print(data)
    print(label)
    break

# output:
# tensor([[-0.0950,  1.7440, -0.4715, -1.4844,  0.3133,  0.3470,  0.9434, -1.1918,
#           0.4125, -0.1721],
#         [ 0.6643,  0.4226, -1.9824,  0.0295, -0.2965,  1.8848,  1.5344, -1.9852,
#           0.2933,  1.6578]])
# tensor([[5],
#         [7]])
# tensor([[ 0.5516,  0.2043,  0.4234, -1.1097, -0.0416, -0.0722,  2.6554,  0.3579,
#           0.6258,  1.1075],
#         [ 1.6676,  0.3645,  1.8899,  1.1604, -0.4062, -0.3100,  1.5496,  0.2190,
#          -0.9531, -0.5275]])
# tensor([[4],
#         [7]])


# 接着我们继续来定义一个简单的多层感知机模型
class MyModel(nn.Module):
    """ 这个模型含有三个线性层，两个激活函数层和一个随机失活层 """
    def __init__(self, in_channel, output_channel, drop_rate=0.0):
        super().__init__()
        # 定义第一个线性层，需要定义线性层的输入维度和输出维度，这里就把输出维度设置为输入维度的四倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel * 4)  # 线性层等价于Keras的Dense层，即全连接层
        # 定义第二个线性层，将输入维度和输出维度设为in_channel的4倍
        self.fc2 = nn.Linear(in_features=in_channel * 4, out_features=in_channel * 4)
        # 定义最后一层的线性层(也可以称为分类层)，用于分类，将数据维度压缩至output_channel的维度用于输出
        self.head = nn.Linear(in_channel * 4, output_channel)

        # 定义一个dropout层用于随机失活神经元
        self.drop1 = nn.Dropout(drop_rate)

        # 定义第一个线性层后的激活函数，处理dropout后的结果
        self.act1 = nn.LeakyReLU()
        # 定义第二个线性层后的激活函数，处理dropout后的结果
        self.act2 = nn.LeakyReLU()
        # 定义softmax函数用于将最终的输出结果映射到(0, 1)之间，dim=1表示softmax在维度为1的数据上进行处理
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """ 这个函数的作用是定义模型前向计算，指定了如何根据输入数据 x 计算出模型的输出结果，是定义模型结构的关键
        在__init__函数中，我们只定义了我们所需要的网络层，并没有定义数据是先经过哪个网络层再经过哪个网络层。
        模型前向计算也可以看作数据 x 的流动方向，先从哪里流向哪里
        """
        # 输入时，x.shape=(batch_size, 10)
        x = self.fc1(x)  # 先将 x 输入至第一个线性层中，输出维度扩大为输入维度的4倍，此时 x.shape=(batch_size, 40)
        x = self.drop1(x)  # dropout并不会改变 x 的形状，所以 x.shape=(batch_size, 40)
        x = self.act1(x)  # 同样LeakyReLU也不会改变 x 的形状，所以 x.shape=(batch_size, 40)
        x = self.fc2(x)  # 第二个线性层的输入维度和输出维度一致，所以 x 的形状保持不变 x.shape=(batch_size, 40)
        x = self.act2(x)  # 同理 x.shape=(batch_size, 40)
        output = self.head(x)  # x 输入至最后一层分类层，数据的输出维度为output_channel，因此 output.shape=(batch_size, 10)
        output = self.softmax(output)  # softmax同样不改变数据的形状，因此 output.shape=(batch_size, 10)
        return output  # 最后将模型的计算结果返回


from torchinfo import summary
model = MyModel(10, 10, drop_rate=0.1)
summary(model, input_size=[(2, 10)], dtypes=[torch.float32], col_names=["input_size", "output_size", "num_params"])
# print(model)
# print(model(data))
#
# output:
# ===================================================================================================================
# Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
# ===================================================================================================================
# MyModel                                  [2, 10]                   [2, 10]                   --
# ├─Linear: 1-1                            [2, 10]                   [2, 40]                   440
# ├─Dropout: 1-2                           [2, 40]                   [2, 40]                   --
# ├─LeakyReLU: 1-3                         [2, 40]                   [2, 40]                   --
# ├─Linear: 1-4                            [2, 40]                   [2, 40]                   1,640
# ├─LeakyReLU: 1-5                         [2, 40]                   [2, 40]                   --
# ├─Linear: 1-6                            [2, 40]                   [2, 10]                   410
# ├─Softmax: 1-7                           [2, 10]                   [2, 10]                   --
# ===================================================================================================================
# Total params: 2,490
# Trainable params: 2,490
# Non-trainable params: 0
# Total mult-adds (M): 0.00
# ===================================================================================================================
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.00
# Params size (MB): 0.01
# Estimated Total Size (MB): 0.01
# ===================================================================================================================


for data, label in loader:
    print('data:', data)
    print('label:', label)
    print('result:', model(data))
    break

# ouptut:
# data: tensor([[ 4.2791e-03,  2.1216e-01, -6.4535e-01,  1.7263e-01, -4.0761e-02,
#           5.4680e-01,  8.8593e-01, -6.5681e-01,  9.6707e-01, -1.7215e-01],
#         [-4.4312e-04, -2.1998e+00, -7.8363e-01, -3.5219e-01, -1.0839e+00,
#           1.7476e+00,  4.2965e-02,  6.5142e-01, -7.7103e-01, -1.2651e+00]])
# label: tensor([[4],
#         [8]])
# result: tensor([[0.0977, 0.0960, 0.1146, 0.1123, 0.0823, 0.0900, 0.0949, 0.0987, 0.1146,
#          0.0989],
#         [0.0953, 0.0918, 0.1067, 0.0984, 0.0804, 0.1040, 0.1017, 0.1011, 0.1103,
#          0.1104]], grad_fn=<SoftmaxBackward0>)

# 交叉熵是最常用的分类损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器可以使用Adam，需要输入模型的参数和学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练代码可以写成函数的形式，也可以直接写，这里用直接写训练过程的方式来展示
# 首先要定义我们需要训练多少轮，batch_size要定义为多少
EPOCHS = 2
BATCH_SIZE = 2
# 其次，torch的训练不会自动把数据和模型加载进GPU中，所以需要我们定义一个训练设备，例如device
device = torch.device('cpu')  # 前期学习就只使用CPU训练
# ！！重点，如果是使用GPU进行训练，那么我们需要把模型也加载进GPU中，不然就无法使用GPU训练
model = model.to(device)  # 把模型加载至训练设备中

# 定义一个含有4000条数据的训练集和1000条数据的验证集
train_dataset = MyDataset(4000)
valid_dataset = MyDataset(1000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 需要注意的是，验证集的dataloader不需要打乱数据


# for epoch in range(EPOCHS):
#
#     # -------------------------------------------------训练代码开始-------------------------------------------------#
#     size = len(train_dataset)  # 获取训练集的长度，也就是数据量的大小，用于输出中间结果时评估训练进度
#     # 模型有两种模式，train模式下模型会计算梯度用于反向传播，推理速度较慢；在eval模式下模型不会计算梯度，可以大大加快模型的推理速度。
#     model.train()  # 将模型调整为训练模式
#     # 这里采用枚举的方式读取数据，这样就可以用 batch 来当做标识，每 100 个batch就输出一次模型的训练结果
#     for batch, (X, y) in enumerate(train_loader):
#         # -------------------------------------------模型推理开始（基本）--------------------------------------------#
#         X, y = X.to(device), y.to(device)  # 将数据和标签都加载至训练设备中
#         pred = model(X)  # 将 X 输入至模型中计算结果
#         # -------------------------------------------模型推理结束（基本）--------------------------------------------#
#
#         # -------------------------------------------计算损失开始（基本）--------------------------------------------#
#         loss = loss_fn(pred, y)  # 用损失函数对模型的预测结果和标签进行计算
#         # -------------------------------------------计算损失结束（基本）--------------------------------------------#
#
#         # -------------------------------------------反向传播开始（基本）--------------------------------------------#
#         optimizer.zero_grad()  # 首先需要将优化器的梯度初始化为0，如果没有初始化，之前每个batch计算的梯度就会累积起来
#         loss.backward()  # 之后损失函数计算输出的梯度(误差)，同时将梯度从输出层向输入层反向传播，并通过链式法则计算每个神经元的梯度
#         optimizer.step()  # 最后根据梯度下降法计算损失函数关于每个参数的梯度，并更新模型的参数
#         # -------------------------------------------反向传播结束（基本）--------------------------------------------#
#
#         # -----------------------------------------输出中间结果开始（可替代）-----------------------------------------#
#         if batch % 100 == 0:  # 每100个 batch 就输出当前的损失值和已经用于训练的数据量
#             # loss.item() 可以将loss中的损失函数值取出，(batch + 1) * len(X)表示目前已经训练了多少的数据
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 输出该batch的损失和训练进度
#         # -----------------------------------------输出中间结果开始（可替代）-----------------------------------------#
#     # -------------------------------------------------训练代码结束-------------------------------------------------#
#
#     # 本轮训练完毕后，如果有验证集的话，我们需要编写验证代码来验证本轮模型的训练效果
#     # -------------------------------------------------验证代码开始-------------------------------------------------#
#     model.eval()  # 将模型调整为验证模式，加快推理速度
#     size = len(valid_dataset)  # 用于最后统计准确率，这个可加可不加
#     num_batches = len(valid_loader)  # 用于统计每个batch的损失，这个也是可加可不加
#     test_loss, correct = 0, 0  # 最终输出的准确率和平均损失
#     with torch.no_grad():  # 在验证时使用，使得模型在推理过程中不计算梯度，大大加快推理速度
#         for X, y in valid_loader:  # 因为在验证过程中并不需要观察验证中间过程的损失值等，所以不需要使用枚举，直接循环
#             # -----------------------------------------模型推理开始（基本）------------------------------------------#
#             X, y = X.to(device), y.to(device)  # 同样也需要把数据和标签都加载至训练设备中
#             pred = model(X)  # 将 X 输入至模型中进行推理，计算预测结果
#             # -----------------------------------------模型推理结束（基本）------------------------------------------#
#
#             # ---------------------------------------统计模型结果开始（可替代）---------------------------------------#
#             test_loss += loss_fn(pred, y).item()  # 将预测结果和标签的损失作为当前batch的损失，加至test_loss中
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 将预测正确的数量，加至correct中
#             # ---------------------------------------统计模型结果结束（可替代）---------------------------------------#
#
#     test_loss /= num_batches  # 对所有batch的损失求平均，得到每个batch的平均损失
#     correct /= size  # 计算预测正确的样本数量占所有样本的百分比，作为准确率
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     # -------------------------------------------------验证代码结束-------------------------------------------------#


# 如果要写成函数的形式怎么写呢？
def train_one_epoch(dataset, dataloader, model, loss_fn, optimizer):
    # -------------------------------------------------训练代码开始-------------------------------------------------#
    size = len(dataset)  # 获取训练集的长度，也就是数据量的大小，用于输出中间结果时评估训练进度
    # 模型有两种模式，train模式下模型会计算梯度用于反向传播，推理速度较慢；在eval模式下模型不会计算梯度，可以大大加快模型的推理速度。
    model.train()  # 将模型调整为训练模式
    # 这里采用枚举的方式读取数据，这样就可以用 batch 来当做标识，每 100 个batch就输出一次模型的训练结果
    for batch, (X, y) in enumerate(dataloader):
        # -------------------------------------------模型推理开始（基本）--------------------------------------------#
        X, y = X.to(device), y.to(device)  # 将数据和标签都加载至训练设备中
        pred = model(X)  # 将 X 输入至模型中计算结果
        # -------------------------------------------模型推理结束（基本）--------------------------------------------#

        # -------------------------------------------计算损失开始（基本）--------------------------------------------#
        loss = loss_fn(pred, y)  # 用损失函数对模型的预测结果和标签进行计算
        # -------------------------------------------计算损失结束（基本）--------------------------------------------#

        # -------------------------------------------反向传播开始（基本）--------------------------------------------#
        optimizer.zero_grad()  # 首先需要将优化器的梯度初始化为0，如果没有初始化，之前每个batch计算的梯度就会累积起来
        loss.backward()  # 之后损失函数计算输出的梯度(误差)，同时将梯度从输出层向输入层反向传播，并通过链式法则计算每个神经元的梯度
        optimizer.step()  # 最后根据梯度下降法计算损失函数关于每个参数的梯度，并更新模型的参数
        # -------------------------------------------反向传播结束（基本）--------------------------------------------#

        # -----------------------------------------输出中间结果开始（可替代）-----------------------------------------#
        if batch % 100 == 0:  # 每100个 batch 就输出当前的损失值和已经用于训练的数据量
            # loss.item() 可以将loss中的损失函数值取出，(batch + 1) * len(X)表示目前已经训练了多少的数据
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 输出该batch的损失和训练进度
        # -----------------------------------------输出中间结果开始（可替代）-----------------------------------------#
    # -------------------------------------------------训练代码结束-------------------------------------------------#


def valid_one_epoch(dataset, dataloader, model, loss_fn):
    # 本轮训练完毕后，如果有验证集的话，我们需要编写验证代码来验证本轮模型的训练效果
    # -------------------------------------------------验证代码开始-------------------------------------------------#
    model.eval()  # 将模型调整为验证模式，加快推理速度
    size = len(dataset)  # 用于最后统计准确率，这个可加可不加
    num_batches = len(dataloader)  # 用于统计每个batch的损失，这个也是可加可不加
    test_loss, correct = 0, 0  # 最终输出的准确率和平均损失
    with torch.no_grad():  # 在验证时使用，使得模型在推理过程中不计算梯度，大大加快推理速度
        for X, y in dataloader:  # 因为在验证过程中并不需要观察验证中间过程的损失值等，所以不需要使用枚举，直接循环
            # -----------------------------------------模型推理开始（基本）------------------------------------------#
            X, y = X.to(device), y.to(device)  # 同样也需要把数据和标签都加载至训练设备中
            pred = model(X)  # 将 X 输入至模型中进行推理，计算预测结果
            # -----------------------------------------模型推理结束（基本）------------------------------------------#

            # ---------------------------------------统计模型结果开始（可替代）---------------------------------------#
            test_loss += loss_fn(pred, y).item()  # 将预测结果和标签的损失作为当前batch的损失，加至test_loss中
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 将预测正确的数量，加至correct中
            # ---------------------------------------统计模型结果结束（可替代）---------------------------------------#

    test_loss /= num_batches  # 对所有batch的损失求平均，得到每个batch的平均损失
    correct /= size  # 计算预测正确的样本数量占所有样本的百分比，作为准确率
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # -------------------------------------------------验证代码结束-------------------------------------------------#


# for epoch in range(EPOCHS):
#     print(f"Epoch {epoch+1} / {EPOCHS}\n-------------------------------")
#     train_one_epoch(train_dataset, train_loader, model, loss_fn, optimizer)
#     valid_one_epoch(valid_dataset, valid_loader, model, loss_fn)

# output:
# Epoch 1 / 10
# -------------------------------
# loss: 2.313363  [    2/ 4000]
# loss: 2.305264  [  202/ 4000]
# loss: 2.294294  [  402/ 4000]
# loss: 2.295321  [  602/ 4000]
# loss: 2.309459  [  802/ 4000]
# loss: 2.303990  [ 1002/ 4000]
# loss: 2.284616  [ 1202/ 4000]
# loss: 2.300499  [ 1402/ 4000]
# loss: 2.311989  [ 1602/ 4000]
# loss: 2.304394  [ 1802/ 4000]
# loss: 2.298525  [ 2002/ 4000]
# loss: 2.281807  [ 2202/ 4000]
# loss: 2.294565  [ 2402/ 4000]
# loss: 2.313917  [ 2602/ 4000]
# loss: 2.297346  [ 2802/ 4000]
# loss: 2.302145  [ 3002/ 4000]
# loss: 2.309603  [ 3202/ 4000]

# # 模型训练完毕之后，我们要把模型保存下来
# # 注意！在模型训练过程中，模型会保留中间状态，例如 Batch Normalization 中的均值和方差等等，
# # 而在验证时，这些中间状态是不需要的，因为每个测试样本只需要使用它自己的信息。
# # 因此，在测试过程中，需要将模型的状态切换为评估模式（eval mode），以确保中间状态不会影响测试结果。
# model.eval()  # 首先切换到评估模式(eval mode)
#
# # torch的模型保存分两种情况，一种是只保存模型的权重，不保存模型结构；另一种是把模型结构也保存，文件会大一点点。
#
# # 仅保存模型的权重，需要先定义好模型结构后才能通过load_state_dict的方法载入模型权重
# torch.save(model.state_dict(), "model_checkpoint.pth")
# model.load_state_dict(torch.load('model_checkpoint.pth'))
#
# # 保存包括模型结构的全部信息，可以通过load的方法直接加载整个模型结构和权重
# torch.save(model, "model.pth")
# model2 = torch.load('model.pth')
#
# # 可以通过loader中的数据来判断两个模型的结果是否一致
# for data, label in loader:
#     print('data:', data)
#     print('label:', label)
#     print('result1:', model(data))
#     print('result2:', model2(data))
#     break

# output：
# data: tensor([[ 0.4659,  1.5091, -0.2391, -1.0837, -1.5378,  0.8773,  0.6433,  1.4609,
#           0.4303, -1.3269],
#         [-0.9014,  0.3153, -0.1788,  1.4997, -3.0364,  0.7569, -1.4344, -1.0127,
#           0.6083, -0.2906]])
# label: tensor([7, 5])
# result1: tensor([[0.0948, 0.1421, 0.0982, 0.0938, 0.1013, 0.0861, 0.1041, 0.0766, 0.1067,
#          0.0962],
#         [0.0876, 0.1373, 0.1028, 0.0869, 0.1022, 0.0754, 0.1170, 0.0945, 0.1009,
#          0.0953]], grad_fn=<SoftmaxBackward0>)
# result2: tensor([[0.0948, 0.1421, 0.0982, 0.0938, 0.1013, 0.0861, 0.1041, 0.0766, 0.1067,
#          0.0962],
#         [0.0876, 0.1373, 0.1028, 0.0869, 0.1022, 0.0754, 0.1170, 0.0945, 0.1009,
#          0.0953]], grad_fn=<SoftmaxBackward0>)


from tqdm import tqdm


def simple_train(dataloader, model, loss_fn, optimizer):
    model.train()  # 将模型调整为训练模式
    average_loss = 0.  # 定义平均损失为0.
    train_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for batch, (X, y) in train_bar:
        X, y = X.to(device), y.to(device)  # 将数据和标签都加载至训练设备中
        pred = model(X)  # 将 X 输入至模型中计算结果
        loss = loss_fn(pred, y)  # 用损失函数对模型的预测结果和标签进行计算
        average_loss += loss.item()  # 把损失加到平均损失中，可以用在进度条上实时显示该batch的损失

        optimizer.zero_grad()  # 首先需要将优化器的梯度初始化为0，如果没有初始化，之前每个batch计算的梯度就会累积起来
        loss.backward()  # 之后损失函数计算输出的梯度(误差)，同时将梯度从输出层向输入层反向传播，并通过链式法则计算每个神经元的梯度
        optimizer.step()  # 最后根据梯度下降法计算损失函数关于每个参数的梯度，并更新模型的参数

        # set_postfix可以给进度条的后面加上后缀，参数名称写什么进度条的后面就会显示什么
        train_bar.set_postfix(loss=average_loss / (batch + 1))
        train_bar.update()  # 立即更新进度条，方便看到训练进度


def simple_valid(dataloader, model, best_score):
    model.eval()  # 将模型调整为验证模式，加快推理速度
    pred_list, target_list = [], []
    with torch.no_grad():  # 在验证时使用，使得模型在推理过程中不计算梯度，大大加快推理速度
        for X, y in dataloader:  # 因为在验证过程中并不需要观察验证中间过程的损失值等，所以不需要使用枚举，直接循环
            X, y = X.to(device), y.to(device)  # 同样也需要把数据和标签都加载至训练设备中
            pred = model(X)  # 将 X 输入至模型中进行推理，计算预测结果

            # 将每个样本的预测结果概率中最大值的索引当成预测结果取出
            # 如果是用GPU训练的话还要加上.cpu()表示把数据传输到cpu上，因为在GPU上没法处理；之后再用.numpy()表示只取预测结果的数值
            pred_list.append(pred.argmax(1).cpu().numpy())  # 预测结果列表
            target_list.append(y.cpu().numpy())  # 标签列表
        # 预测结果列表和标签列表都是按batch拼接的，因此我们需要使用numpy转换成按样本拼接，这样才能计算混淆矩阵和分类报告
        predictions = np.concatenate(pred_list, axis=0)
        targets = np.concatenate(target_list, axis=0)
        # 接下来就可以把标签和预测结果输入函数中啦，记得不要搞反了~ 还有labels建议也要输入，否则混淆矩阵和分类报告的行列都会随机，不方便统计结果
        # print(confusion_matrix(targets, predictions, labels=range(0, 10)))
        # print(classification_report(targets, predictions, labels=range(0, 10)))
        f1 = f1_score(targets, predictions, average='macro', labels=range(0, 10))

        if f1 > best_score:
            print(f'------ The best result improved from {best_score} to {f1} -----')
            best_score = f1
            torch.save(model.state_dict(), 'best.pth')
        else:
            if os.path.exists(f'best.pth'):
                model.load_state_dict(torch.load('best.pth'))
                print(f'model restore the best.pth')
            print(f'best m_score till now: {best_score}')
    return best_score


best_score = 0.
for epoch in range(EPOCHS * 20):
    print(f"Epoch {epoch+1} / {EPOCHS * 20}\n-------------------------------")
    simple_train(train_loader, model, loss_fn, optimizer)
    best_score = simple_valid(valid_loader, model, best_score)

# output:
# Epoch 1 / 2
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 963.73it/s, loss=2.3]
# [[31  1  0 16  7 32  1  0  8  0]
#  [33  0  0  9  9 41  0  0  5  0]
#  [24  0  0 24  6 34  2  0  8  0]
#  [31  1  0 10  7 38  3  0 12  0]
#  [28  2  0 16  7 40  0  0 12  0]
#  [28  1  0 15  6 34  0  0 11  0]
#  [21  0  0 17  8 25  1  0  5  0]
#  [33  3  0 19  8 42  1  0 13  0]
#  [33  1  0 14 10 38  1  0  6  0]
#  [23  0  0 17 11 44  3  0 10  0]]
#               precision    recall  f1-score   support
#
#            0       0.11      0.32      0.16        96
#            1       0.00      0.00      0.00        97
#            2       0.00      0.00      0.00        98
#            3       0.06      0.10      0.08       102
#            4       0.09      0.07      0.08       105
#            5       0.09      0.36      0.15        95
#            6       0.08      0.01      0.02        77
#            7       0.00      0.00      0.00       119
#            8       0.07      0.06      0.06       103
#            9       0.00      0.00      0.00       108
#
#     accuracy                           0.09      1000
#    macro avg       0.05      0.09      0.05      1000
# weighted avg       0.05      0.09      0.05      1000
#
# Epoch 2 / 2
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 879.49it/s, loss=2.3]
# [[29  1  0  7 13 29  7  0 10  0]
#  [30  0  0  6  5 37 12  0  7  0]
#  [20  0  0 11 14 33  9  0 11  0]
#  [29  0  0  3  4 37 15  0 14  0]
#  [29  0  0  5 10 38  9  0 14  0]
#  [26  1  0  7  7 33  8  0 13  0]
#  [23  0  0  9  8 23  9  0  5  0]
#  [34  1  0  7 11 40  9  0 17  0]
#  [27  1  0  8  8 37 13  0  9  0]
#  [23  0  0 11 11 44  7  0 12  0]]
#               precision    recall  f1-score   support
#
#            0       0.11      0.30      0.16        96
#            1       0.00      0.00      0.00        97
#            2       0.00      0.00      0.00        98
#            3       0.04      0.03      0.03       102
#            4       0.11      0.10      0.10       105
#            5       0.09      0.35      0.15        95
#            6       0.09      0.12      0.10        77
#            7       0.00      0.00      0.00       119
#            8       0.08      0.09      0.08       103
#            9       0.00      0.00      0.00       108
#
#     accuracy                           0.09      1000
#    macro avg       0.05      0.10      0.06      1000
# weighted avg       0.05      0.09      0.06      1000

# output:
# Epoch 1 / 40
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 978.63it/s, loss=2.3]
# ------ The best result improved from 0.0 to 0.03416803732323702 -----
# Epoch 2 / 40
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 963.66it/s, loss=2.3]
# ------ The best result improved from 0.03416803732323702 to 0.03607081757839706 -----
# Epoch 3 / 40
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 884.89it/s, loss=2.3]
# model restore the best.pth
# best m_score till now: 0.03607081757839706
# Epoch 4 / 40
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 857.29it/s, loss=2.3]
# model restore the best.pth
# best m_score till now: 0.03607081757839706
# Epoch 5 / 40
# -------------------------------
# Train: 100%|██████████| 2000/2000 [00:02<00:00, 811.94it/s, loss=2.3]
# model restore the best.pth
# best m_score till now: 0.03607081757839706
