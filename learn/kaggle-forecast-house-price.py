import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import d2l.torch as d2l
import matplotlib as plt

data_dir = 'D:\备份\研究生成果/0.Dataset\house-prices-advanced-regression-techniques'

train_data = pd.read_csv(os.path.join(data_dir,'train.csv'))
test_data = pd.read_csv(os.path.join(data_dir,'test.csv'))

# print(train_data.shape,'\n',test_data.shape)

all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:-1]))

# print(all_features.shape)

#区分数值特征和非数值特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

#将所有数值特征归一化
# lambda 匿名函数  :前面是输入后面是输出
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))

#将所有数值特征中的NaN填充为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#将非数值特征one-hot编码
all_features = pd.get_dummies(all_features,dummy_na=True)
# print(all_features.shape)

n_train =train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
                              dtype = torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype = torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.T,
                            dtype=torch.float32)

loss = nn.MSELoss()

in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net,features,labels):
    clipped_preds = torch.clamp(net(features),1,float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds).reshape(-1),torch.log((labels))))
    return rmse.item()

def train(net,train_feature,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = d2l.load_array((train_feature,train_labels),batch_size)
    optimizer = torch.optim.Adam(net.parameters(),learning_rate,weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for x,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x).reshape(-1),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_feature,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size = x.shape[0] // k
    x_train,y_train =None,None
    for j in range(k):
        idx = slice(j * fold_size,(j+1)*fold_size)
        x_part,y_part = x[idx,:],y[idx]
        if j == i:
            x_valid, y_valid =x_part ,y_part
        elif x_train is None:
            x_train,y_train = x_part,y_part
        else:
            x_train = torch.cat([x_train,x_part],0)
            y_train = torch.cat([y_train,y_part],0)
    return x_train,y_train,x_valid,y_valid

def k_fold(k,x_train,y_train,num_epoch,learning_rate,weight_dacy,batch_size):
    train_l_sum,valid_l_sum = 0,0
    for i in range(k):
        data = get_k_fold_data(k,i,x_train,y_train)
        net = get_net()
        train_ls ,valid_ls =train(net,*data,num_epoch,learning_rate,weight_dacy,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1,num_epoch + 1)),[train_ls,valid_ls],xlabel='epoch',ylabel='rmse',xlim=[1,num_epoch+1],legend=['train','test'],yscale='log')
        print(f'fold{i+1},train log rmse {float(train_ls[-1]):f}','        ',
              f'valid log rmse{float(valid_ls[-1]):f}')
    return  train_l_sum/k,valid_l_sum/k


k,num_epochs,lr ,weight_decay,batch_size =5,100,5,0,64
#k折交叉验证调参
# train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
# print(f'{k}-折验证:\n平均训练 log rmse {float(train_l):f}\n'
#       f'平均验证 log rmse{float(valid_l):f}')

def train_and_pred(train_feature,test_feature,train_labels,test_data,num_epoch,lr,weight_decay,batch_size):
    net = get_net()
    train_ls, _ = train(net,train_feature,train_labels,None,None,num_epoch,lr,weight_decay,batch_size)
    d2l.plot(np.arange(1,num_epoch+1),[train_ls],xlabel='epcch',ylabel='log rmse',xlim=[1,num_epoch],yscale='log')
    print(f'train log rmse{float(train_ls[-1]):f}')
    preds = net(test_feature).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.T[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submisiion.csv',index=False)

train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)