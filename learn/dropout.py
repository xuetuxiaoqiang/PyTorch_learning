import torch
import torch.nn as nn
import d2l.torch as d2l
def dropout_layer(X,dropout):
    assert 0<= dropout <=1
    if dropout ==1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand_like(X)>dropout).float()
    return mask * X / (1.0 - dropout)

#dropout_layer 运行
# x = torch.arange(16,dtype= torch.float32).reshape((2,8))
# print(x)
# print(dropout_layer(x,0.))
# print(dropout_layer(x,0.5))
# print(dropout_layer(x,1))

num_input,num_output,num_hidden1,num_hidden2 = 784,10,256,256
dropout1,dropout2 = 0.2 ,0.5

class Net (nn.Module):
    def __init__(self,num_input,num_output,num_hidden1,num_hidden2,is_training=True):
        super(Net,self).__init__()
        self.num_inputs = num_input
        self.training = is_training
        self.lin1 = nn.Linear(num_input,num_hidden1)
        self.lin2 = nn.Linear(num_hidden1,num_hidden2)
        self.lin3 = nn.Linear(num_hidden2,num_output)
        self.relu = nn.ReLU()

    def forward (self,x):
        H1 = self.relu(self.lin1(x.reshape(-1,self.num_inputs)))
        if self.training:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out



num_epoch,lr,batch_size = 10 ,0.5,256

train_iter ,test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')

#从零开始实现
# net = Net(num_input,num_output,num_hidden1,num_hidden2)
# optimer = torch.optim.SGD(net.parameters(),lr)
# d2l.train_ch3(net,train_iter,test_iter,loss,num_epoch,optimer)


#pytorch实现

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_input,num_hidden1),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(num_hidden1,num_hidden2),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(num_hidden2,num_output)
                    )

net.apply(init_weights)
optimer = torch.optim.SGD(net.parameters(),lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epoch,optimer)

