import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn
from IPython import display
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from d2l.torch import get_fashion_mnist_labels,show_images,get_dataloader_workers

d2l.use_svg_display()

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root='D:\备份\研究生成果/0.Dataset',train=True,transform=trans)
mnist_test  = torchvision.datasets.FashionMNIST(root='D:\备份\研究生成果/0.Dataset',train=False,transform=trans)

#可视化数据图片
# x,y =next(iter(data.DataLoader(mnist_train,batch_size=18)))
# show_images(x.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))
#
# d2l.plt.show()

batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())
test_iter  = data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_dataloader_workers())

#数据加载时间
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f}sec')

num_input = 784
num_output = 10


w = torch.normal (0,0.01,size=(num_input,num_output),requires_grad=True)
b = torch.zeros(num_output,requires_grad=True)

#矩阵运算dim表示要要运算的维度
# X = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
# print(X.sum(dim=0,keepdim = True))
# print(X.sum(dim=1,keepdim = True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition

#softmax 演示
# X = torch.normal(0,1,(2,5))
# X_prob = softmax(X)
# print(X,'\n',X_prob,'\n',X_prob.sum(1))

def softmax_layer(X):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b)

#softmax_layer 演示
# X=torch.randn((2,28,28))
# print(softmax_layer(X),'\n',softmax_layer(X).sum(1))

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

y = torch.tensor([0,2,1])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5],[0.6,0.1,0.3]])

#交叉熵损失演示
# print(y_hat[[0,1],y])
# print(cross_entropy(y_hat,y))

def accuracy(y_hat , y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())     #cmp.count_nonzero()也可以用

#accuracy演示
print(accuracy(y_hat,y)/len(y))

def evalutate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x,y in data_iter:
        metric.add(accuracy(net(x),y),y.numel())
    return metric[0]/metric[1]

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n

    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#Accumulator 演示
# print(evalutate_accuracy(softmax_layer,train_iter))

def train_epoch_ch3(net, train_iter, loss ,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y),
                accuracy(y_hat,y),
                y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),
                       accuracy(y_hat,y),
                       y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

num_epochs = 5

def train_ch3 (net,train_iter,test_iter,loss ,num_epochs,updater,
               animator=Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend=['train loss','train acc','test acc'])):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evalutate_accuracy(net,test_iter)
        animator.add(epoch+1,train_metrics+(test_acc,))
    train_loss ,train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater (batch_size):
    return d2l.sgd([w,b],lr,batch_size)
#整体训练
# train_ch3(softmax_layer,train_iter,test_iter,cross_entropy,num_epochs,updater)

def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    print(trues,'\n',preds)
    titles = [true + '/n' + pred for true , pred in zip(trues,preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])

#整体预测
# predict_ch3(softmax_layer,test_iter)
# plt.show()