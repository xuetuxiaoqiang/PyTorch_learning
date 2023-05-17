import math

import torch.optim

from d2l import torch as d2l
import numpy as np
import torch.nn as nn

max_degree = 20
n_train, n_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train + n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale=0.1,size=labels.shape)

true_w,features,poly_features,labels = [torch.tensor(x,dtype=torch.float32)for x in [true_w,features,poly_features,labels]]


def evaluate_loss(net, data_iter,loss):
    metric = d2l.Accumulator(2)
    for x,y in data_iter:
        out = net(x)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]

def train(train_features,test_features,train_labels,test_labels,num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
    print(train_features.shape)
    print(train_labels.shape)
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size,is_train=False)
    optimer = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[1,num_epochs],ylim=[1e-3,1e-1],legend=['train','test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net,train_iter,loss,optimer)
        if epoch == 0 or (epoch+1)%20 == 0:
            animator.add(epoch+1,(evaluate_loss(net,train_iter,loss),evaluate_loss(net,test_iter,loss)))
    print('weight:',net[0].weight.data.numpy())

#正常训练
# train(poly_features[:n_train,:4],poly_features[n_train:,:4],labels[:n_train],labels[n_train:])
#欠拟合
# train(poly_features[:n_train,:2],poly_features[n_train:,:2],labels[:n_train],labels[n_train:])
#过拟合
train(poly_features[:n_train,:],poly_features[n_train:,:],labels[:n_train],labels[n_train:])