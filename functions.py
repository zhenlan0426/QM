#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:57:40 2019

@author: will
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU,BatchNorm1d,Dropout,RReLU
from torch_geometric.nn import NNConv
import numpy as np

def Rx():
    theta = torch.rand(1)*np.pi*2
    return torch.tensor([[1,0,0],\
                         [0,torch.cos(theta),-torch.sin(theta)],\
                         [0,torch.sin(theta),torch.cos(theta)]])
def Ry():
    theta = torch.rand(1)*np.pi*2
    return torch.tensor([[torch.cos(theta),0,torch.sin(theta)],\
                         [0,1,0],\
                         [-torch.sin(theta),0,torch.cos(theta)]])
def Rz():
    theta = torch.rand(1)*np.pi*2
    return torch.tensor([[torch.cos(theta),-torch.sin(theta),0],\
                         [torch.sin(theta),torch.cos(theta),0],\
                         [0,0,1]])
def R():
    # rotation transform
    return torch.matmul(torch.matmul(Rz(),Ry()),Rx())

def transform_xyz(node):
    x = node['x']
    x[:,:3] = torch.matmul(R(),x[:,:3].t()).t()
    return node


class Net(torch.nn.Module):
    def __init__(self,dim=64,edge_dim=12):
        super(Net, self).__init__()
        self.lin_node = torch.nn.Linear(8, dim)
        self.lin_edge_attr = torch.nn.Linear(19, edge_dim)
        
        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)
        self.lin1 = Sequential(Linear(dim+8, 128), ReLU(), Linear(128, 1))
        
    def forward(self, data,IsTrain=False):
        out = F.relu(self.lin_node(data.x))
        edge_attr = F.relu(self.lin_edge_attr(data.edge_attr))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            
        yhat = torch.cat([out[data.edge_index3].mean(0),data.edge_attr3],1)
        yhat = self.lin1(yhat).squeeze(1)  
        
        if IsTrain:
            k = torch.sum(data.edge_attr3,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(data.y-yhat).unsqueeze(1)
            loss = torch.sum(torch.log(torch.sum(abs_ * data.edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex]))/nonzeroIndex.shape[0]
            return loss
        else:
            return yhat
        
class Net_int(torch.nn.Module):
    # differ from Net in that edge_attr3 is used to determine Weight
    def __init__(self,dim=64,edge_dim=12):
        super(Net_int, self).__init__()
        self.lin_node = torch.nn.Linear(8, dim)
        self.lin_edge_attr = torch.nn.Linear(19, edge_dim)
        
        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)
        self.lin_weight = Linear(8, dim*3, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*3)
        
    def forward(self, data,IsTrain=False,logloss=True):
        out = F.relu(self.lin_node(data.x))
        edge_attr = F.relu(self.lin_edge_attr(data.edge_attr))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        temp = out[data.edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)
        yhat = self.norm(yhat)
        weight = self.lin_weight(data.edge_attr3)
        bias = self.lin_bias(data.edge_attr3)
        yhat = torch.sum(yhat * weight,1,keepdim=True) + bias
        yhat = yhat.squeeze(1)
        
        if IsTrain:
            if logloss:
                k = torch.sum(data.edge_attr3,0)
                nonzeroIndex = torch.nonzero(k).squeeze(1)
                abs_ = torch.abs(data.y-yhat).unsqueeze(1)
                loss = torch.sum(torch.log(torch.sum(abs_ * data.edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex]))/nonzeroIndex.shape[0]
                return loss
            else:
                return torch.mean(torch.abs(data.y-yhat))
        else:
            return yhat
        
class Net_int_2Edges(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12):
        super(Net_int, self).__init__()
        self.lin_node = torch.nn.Linear(8, dim)
        self.lin_edge_attr = torch.nn.Linear(19, edge_dim)
        
        nn1 = Linear(edge_dim, dim * dim, bias=False)
        nn2 = Linear(8, dim * dim * 2 * 2, bias=False)
        
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = NNConv(dim*2, dim*2, nn2, aggr='mean', root_weight=False)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.lin_weight = Linear(8, dim*3, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*3)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(data.x))
        edge_attr = F.rrelu(self.lin_edge_attr(data.edge_attr))
        h = out.unsqueeze(0)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        edge_attr3 = torch.cat([data.edge_attr3,data.edge_attr3],0)
        
        for i in range(2):
            # using bonding as edge
            m = F.rrelu(self.conv1(out, data.edge_index, edge_attr))
            out, h = self.gru1(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        out = self.lin_covert(out)
        for i in range(2):
            # using couping as edge
            m = F.rrelu(self.conv2(out, edge_index3, edge_attr3))
            out, h = self.gru2(m.unsqueeze(0), h)
            out = out.squeeze(0)     
            
        temp = out[data.edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)
        yhat = self.norm(yhat)
        weight = self.lin_weight(data.edge_attr3)
        bias = self.lin_bias(data.edge_attr3)
        yhat = torch.sum(yhat * weight,1,keepdim=True) + bias
        yhat = yhat.squeeze(1)
        
        if IsTrain:
            k = torch.sum(data.edge_attr3,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(data.y-yhat).unsqueeze(1)
            loss = torch.sum(torch.log(torch.sum(abs_ * data.edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex]))/nonzeroIndex.shape[0]
            return loss
        else:
            return yhat           
        
        
        
        
        
        
        
        
        
        
        
        