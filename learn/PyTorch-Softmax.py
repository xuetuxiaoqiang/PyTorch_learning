import torch
from d2l import torch as d2l
import torch.nn as nn
from torchvision import transforms

d2l.use_svg_display()

lr = 0.1
num_epochs = 10
batch_size = 256

trans = transforms.ToTensor()

train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

optimer = torch.optim.SGD(net.parameters(),lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,optimer)