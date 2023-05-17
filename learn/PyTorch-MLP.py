from torchvision import transforms
from d2l import torch as d2l
import torch
import torch.nn as nn


batch_size = 256
num_epochs = 10
lr = 0.1
num_input , num_hidden , num_output = 784,256,10

trans = transforms.ToTensor()

train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_input,num_hidden,bias=True),
                    nn.ReLU(),
                    nn.Linear(num_hidden,num_output,bias=True),
                    nn.ReLU())
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

params = net.parameters()
optimer = torch.optim.SGD(params,lr)
loss = nn.CrossEntropyLoss()

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,optimer)