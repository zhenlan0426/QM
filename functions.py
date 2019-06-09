#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:57:40 2019

@author: will
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv

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