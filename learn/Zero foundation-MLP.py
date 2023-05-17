import torch
from d2l import torch as d2l
import torch.nn as nn

batch_size = 256
num_epochs = 10
lr = 0.1
num_input , num_hidden , num_output = 784,256,10

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

w1 = nn.Parameter(torch.randn(num_input,num_hidden),requires_grad=True)
b1 = nn.Parameter(torch.zeros(num_hidden),requires_grad=True)
w2 = nn.Parameter(torch.randn(num_hidden,num_output),requires_grad=True)
b2 = nn.Parameter(torch.zeros(num_output),requires_grad=True)

params = [w1,b1,w2,b2]

def relu(x):
    a =torch.zeros_like(x)
    return torch.max(x,a)

def net(x):
    x = x.reshape(-1,num_input)
    H = relu(x @ w1 + b1)       #@表示矩阵点乘，*表示对应位置元素相乘
    return H @ w2 + b2

loss = nn.CrossEntropyLoss(reduction='none')

optimer = torch.optim.SGD(params,lr)

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,optimer)