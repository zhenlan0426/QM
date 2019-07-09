#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:47:11 2019

@author: will
"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU,BatchNorm1d,Dropout,RReLU
from torch_geometric.nn import NNConv,GATConv
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset,uniform
from torch_geometric.data import Data,DataLoader

import time
import pickle
import numpy as np
import pandas as pd
import copy

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Data ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''
data_dict = {'../Data/{}_data_ACSF.pickle':{'node_in':89,'edge_in':19,'edge_in4':1},\
             '../Data/{}_data_ACSF_expand.pickle':{'node_in':89,'edge_in':19+25,'edge_in4':1+25}}

columns = ['reuse',
		   'block',
		   'head',
		   'data',
		   'batch_size',
		   'dim',
		   'clip',
		   'layer1',
		   'layer2',
		   'factor']

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Head ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

class Set2Set(torch.nn.Module):
    r"""modify the oringinal Set2Set by allowing to pass in h and q_star
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch, h=None,q_star=None):
        """"""
        batch_size = batch.max().item() + 1
        if h is None:
            h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
                 x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        if q_star is None:
            q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
        
class InteractionNet(torch.nn.Module):
    def __init__(self,IntDim,xDimList,FunList):
        # None in FunList mean identity func
        super(InteractionNet, self).__init__()
        self.dimList = list(zip(xDimList[:-1],xDimList[1:]))
        self.linear = nn.ModuleList([Linear(IntDim,d0*d1+d1) for d0,d1 in self.dimList])
        self.FunList = FunList
        
    def forward(self, int_x,x):
        for (d0,d1),lin,fun in zip(self.dimList,self.linear,self.FunList):
            temp = lin(int_x) # (n,d0*d1+d1)
            bias = temp[:,0:d1]
            weight = temp[:,d1:].reshape(-1,d0,d1)
            if fun is not None:
                x = fun(torch.einsum('npq,np->nq',weight,x) + bias)
            else:
                x = torch.einsum('npq,np->nq',weight,x) + bias
        return x        

class InteractionNet2(torch.nn.Module):
    def __init__(self,IntDim,xDimList,FunList):
        # None in FunList mean identity func
        super(InteractionNet, self).__init__()
        self.dimList = list(zip(xDimList[:-1],xDimList[1:]))
        tot_dim = sum([d0*d1+d1 for d0,d1 in self.dimList])
        self.linear = Linear(IntDim,tot_dim)
        self.FunList = FunList
        
    def forward(self, int_x,x):
        temp = self.linear(int_x)
        cum_d = 0
        for (d0,d1),fun in zip(self.dimList,self.FunList):
            bias = temp[:,cum_d:cum_d+d1]
            weight = temp[:,cum_d+d1:cum_d+d1+d0*d1].reshape(-1,d0,d1)
            cum_d = cum_d+d1+d0*d1
            if fun is not None:
                x = fun(torch.einsum('npq,np->nq',weight,x) + bias)
            else:
                x = torch.einsum('npq,np->nq',weight,x) + bias
        return x

class cat3Head(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4):
        cat_factor = 3
        super(cat3Head, self).__init__()
        self.lin_weight = Linear(edge_in3, dim*cat_factor, bias=False)
        self.lin_bias = Linear(edge_in3, 1, bias=False)
        #self.norm = BatchNorm1d(dim*cat_factor)
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        temp = x[edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)
        #yhat = self.norm(yhat)
        weight = self.lin_weight(edge_attr3)
        bias = self.lin_bias(edge_attr3)
        yhat = torch.sum(yhat * weight,1,keepdim=True) + bias
        yhat = yhat.squeeze(1)
        return yhat

class cat3Head_type(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4):
        cat_factor = 3
        super(cat3Head_type, self).__init__()
        self.linear = Sequential(#BatchNorm1d(dim*cat_factor),
                                 Linear(dim*cat_factor,dim*cat_factor*2),
                                 ReLU(inplace=True),
                                 #BatchNorm1d(dim*cat_factor*2),
                                 Linear(dim*cat_factor*2,1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        temp = x[edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)
        yhat = self.linear(yhat).squeeze(1)
        return yhat

class swapHead_type(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4):
        cat_factor = 2
        super(swapHead_type, self).__init__()
        self.linear = Sequential(#BatchNorm1d(dim*cat_factor),
                                 Linear(dim*cat_factor,dim*cat_factor*2),
                                 ReLU(inplace=True),
                                 #BatchNorm1d(dim*cat_factor*2),
                                 Linear(dim*cat_factor*2,1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        temp = x[edge_index3] # (2,N,d)
        yhat = torch.cat([torch.cat([temp[0],temp[1]],1),torch.cat([temp[1],temp[0]],1)],0)
        yhat = self.linear(yhat).squeeze(1).reshape(2,-1).mean(0)
        return yhat
    
class cat3HeadPool_type(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4,processing_steps=6):
        cat_factor = 5
        super(cat3HeadPool_type, self).__init__()
        self.set2set = Set2Set(dim,processing_steps=processing_steps)
        self.linear = Sequential(#BatchNorm1d(dim*cat_factor),
                                 Linear(dim*cat_factor,dim*cat_factor*2),
                                 ReLU(inplace=True),
                                 #BatchNorm1d(dim*cat_factor*2),
                                 Linear(dim*cat_factor*2,1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        coupling_batch_index = batch[edge_index3[0]]
        pool = self.set2set(x, batch) # (m,d)
        pool = pool[coupling_batch_index] # (n_target,d)
        temp = x[edge_index3] # (2,n_target,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2,pool],1)  
        yhat = self.linear(yhat).squeeze(1)
        return yhat

class set2setHead_type(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4,processing_steps=6,num_layers=2):
        super(set2setHead_type, self).__init__()
        self.head = Set2Set(dim,processing_steps=processing_steps,num_layers=num_layers)
        self.linear = Linear(dim*2,1)
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        n = edge_index3.shape[1]
        range_ = torch.arange(n)
        batch_index = torch.cat([range_,range_]).to('cuda:0')
        temp = x[edge_index3].reshape(2*n,-1) # (2*n_target,d)
        yhat = self.head(temp,batch_index)
        yhat = self.linear(yhat).squeeze(1)
        return yhat
    
'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Conv ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

class NNConv2(MessagePassing):
    r""" use element-wise multiplication as in schnet instead of matrix multiplication
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=False,
                 bias=True,
                 **kwargs):
        super(NNConv2, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo) # (n,d)
        return x_j * weight

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)    
        
        

class schnet_block(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12):
        super(schnet_block, self).__init__()
        
        nn = Sequential(BatchNorm1d(edge_dim),Linear(edge_dim, dim*2),ReLU(), \
                        BatchNorm1d(dim*2),Linear(dim*2, dim))
        self.conv = NNConv2(dim, dim, nn, aggr='mean', root_weight=False)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2),ReLU(), \
                                     BatchNorm1d(dim*2),Linear(dim*2, dim))
        
    def forward(self, x, edge_index, edge_attr):
        m = F.relu(self.conv(x, edge_index, edge_attr))
        m = self.lin_covert(m)
        return x + m


class NNConv_block(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12):
        super(NNConv_block, self).__init__()
        
        nn = Sequential(BatchNorm1d(edge_dim),Linear(edge_dim, dim*dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)
        
    def forward(self, x, edge_index, edge_attr):
        m = F.relu(self.conv(x, edge_index, edge_attr))
        out, _ = self.gru(m.unsqueeze(0), x.unsqueeze(0))
        return out.squeeze(0)

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Main ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''


class GNN(torch.nn.Module):

    def __init__(self,reuse,block,head,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN, self).__init__()
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),ReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),ReLU())
        
        if reuse:
            self.conv1 = block(dim=dim,edge_dim=edge_in)
            self.conv2 = block(dim=dim,edge_dim=edge_in3+edge_in4)
        else:
            self.conv1 = nn.ModuleList([block(dim=dim,edge_dim=edge_in) for _ in range(layer1)])
            self.conv2 = nn.ModuleList([block(dim=dim,edge_dim=edge_in3+edge_in4) for _ in range(layer2)])            
        
        self.head = head(dim,edge_in3,edge_in4)
        
    def forward(self, data,IsTrain=False,typeTrain=False):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        temp_ = torch.cat([data.edge_attr3,data.edge_attr4],1) 
        edge_attr3 = torch.cat([temp_,temp_],0)

        for conv in self.conv1:
            out = conv(out,data.edge_index,data.edge_attr)

        for conv in self.conv2:
            out = conv(out,edge_index3,edge_attr3)    
        
        if typeTrain:
            if IsTrain:
                y = data.y[data.type_attr]
            edge_index3 = data.edge_index3[:,data.type_attr]
            edge_attr3 = data.edge_attr3[data.type_attr]
            edge_attr4 = data.edge_attr4[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3 = data.edge_attr3
            edge_attr4 = data.edge_attr4
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr4,data.batch)
        
        if IsTrain:
            k = torch.sum(edge_attr3,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex])
            loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
            return loss,loss_perType
        else:
            return yhat
        

def get_data(data,batch_size):
    with open(data.format('train'), 'rb') as handle:
        train_data = pickle.load(handle)
    with open(data.format('val'), 'rb') as handle:
        val_data = pickle.load(handle)
    
    train_list = [Data(**d) for d in train_data]
    train_dl = DataLoader(train_list,batch_size,shuffle=True)
    val_list = [Data(**d) for d in val_data]
    val_dl = DataLoader(val_list,batch_size,shuffle=False)
    
    return train_dl,val_dl


def train(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,train_loss_list=None,val_loss_list=None):
    since = time.time()
    
    lossBest = 1e6
    if train_loss_list is None:
        train_loss_list,val_loss_list = [],[]
        epoch0 = 0
    else:
        epoch0 = len(train_loss_list)
        
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        np.random.seed()
        train_loss = 0
        train_loss_perType = np.zeros(8)
        val_loss = 0
        val_loss_perType = np.zeros(8)
        
        for i,data in enumerate(train_dl):
            data = data.to('cuda:0')
            loss,loss_perType = model(data,True,typeTrain)
            loss.backward()
            clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            train_loss_perType += loss_perType.cpu().detach().numpy()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(val_dl):
                data = data.to('cuda:0')
                loss,loss_perType = model(data,True,typeTrain)
                val_loss += loss.item()
                val_loss_perType += loss_perType.cpu().detach().numpy()
        
        # save model
        if loss.item()<lossBest:
            lossBest = loss.item()
            torch.save({'model_state_dict': model.state_dict()},'../Model/tmp.tar')
            
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss/j,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType/j])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType/j)
        
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    
    # load best model
    checkpoint = torch.load('../Model/tmp.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model,train_loss_list,val_loss_list
    
    
def save_results(train_loss_perType,val_loss_perType,reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr='base'):
    epochs = len(train_loss_perType)
    results = pd.DataFrame([[reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor] \
                        for _ in range(epochs)],columns=columns,dtype=str)
    temp = pd.DataFrame(np.concatenate([np.arange(epochs,dtype=np.int)[:,np.newaxis],np.stack(train_loss_perType,0),np.stack(val_loss_perType,0)],1),
             columns=['epochs']+['train_type_{}'.format(i) for i in range(8)] + ['val_type_{}'.format(i) for i in range(8)])
    results = pd.concat([results,temp],1)
    results.to_csv('../Data/results_{}.csv'.\
               format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                for i in [reuse,block,head,data,batch_size,dim,clip,\
                                          layer1,layer2,factor,epochs,postStr]])),\
              index=False)    
    
def save_model(model,opt,reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr='base'):
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epochs':epochs
            }, '../Model/{}.tar'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                    for i in [reuse,block,head,data,batch_size,dim,clip,\
                                          layer1,layer2,factor,epochs,postStr]])))


def make_submission(reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr=''):
    # set up
    model = GNN(reuse,block,head,dim,layer1,layer2,factor,**data_dict[data]).to('cuda:0')
    submission = pd.read_csv('../Data/sample_submission.csv')
    
    for i in range(8):
        # load test data and type_id
        with open(data.format('test').split('pickle')[0][:-1]+'_type_'+str(i)+'.pickle', 'rb') as handle:
            test_data = pickle.load(handle)
        test_list = [Data(**d) for d in test_data]
        test_dl = DataLoader(test_list,batch_size,shuffle=False)
        with open(data.format('test').split('pickle')[0][:-1]+'_id_type_'+str(i)+'.pickle', 'rb') as handle:
            test_id = pickle.load(handle)
    
    
        # load model
        checkpoint = torch.load('../Model/{}.tar'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                            for i in [reuse,block,head,data,batch_size,dim,clip,\
                                                  layer1,layer2,factor,epochs,'type_'+str(i)+postStr]])))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
        # predict
        model.eval()
        yhat_list = []
        with torch.no_grad():
            for data_torch in test_dl:
                data_torch = data_torch.to('cuda:0')
                yhat_list.append(model(data_torch,False,True))
        yhat = torch.cat(yhat_list).cpu().detach().numpy()        
        
        # join
        submit_ = dict(zip(test_id,yhat))
        submission['type_'+str(i)] = submission.id.map(submit_)
    
    # save types results    
    submission.to_csv('../Submission/{}.csv'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                        for i in [reuse,block,head,data,batch_size,dim,clip,\
                                              layer1,layer2,factor,epochs,'all_types'+postStr]])),\
                      index=False)
    
    # save final results for submission
    submission['scalar_coupling_constant'] = submission.iloc[:,2:].mean(1)
    submission = submission[['id','scalar_coupling_constant']]
    
    submission.to_csv('../Submission/{}.csv'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                        for i in [reuse,block,head,data,batch_size,dim,clip,\
                                              layer1,layer2,factor,epochs,'final'+postStr]])),\
                      index=False)