#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:57:40 2019

@author: will
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU,BatchNorm1d,Dropout,RReLU
from torch_geometric.nn import NNConv,GATConv,Set2Set
import torch.nn as nn
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
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        self.lin_edge_attr = torch.nn.Linear(edge_in, edge_dim)
        
        nn1 = Linear(edge_dim, dim * dim, bias=False)
        nn2 = Linear(edge_in3, dim * dim * 2 * 2, bias=False)
        
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = NNConv(dim*2, dim*2, nn2, aggr='mean', root_weight=False)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.lin_weight = Linear(8, dim*3*2, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*3*2)
        self.norm_x = BatchNorm1d(node_in)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
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
        h = out.unsqueeze(0)
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
        
        
        
class Net_int_2Edges_attention(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges_attention, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        
        self.conv1 = GATConv(dim, dim, negative_slope=0.2, dropout=0.1, bias=True)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = GATConv(dim*2, dim*2, negative_slope=0.2, dropout=0.1, bias=True)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.lin_weight = Linear(8, dim*3*2, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*3*2)
        self.norm_x = BatchNorm1d(node_in)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
        h = out.unsqueeze(0)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        
        for i in range(2):
            # using bonding as edge
            m = F.rrelu(self.conv1(out, data.edge_index))
            out, h = self.gru1(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        out = self.lin_covert(out)
        h = out.unsqueeze(0)
        for i in range(2):
            # using couping as edge
            m = F.rrelu(self.conv2(out, edge_index3))
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
        
        
        
class Net_int_2Edges_attention2(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges_attention2, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        
        self.conv1 = GATConv(dim, dim, negative_slope=0.2, bias=True)
        
        self.lin_covert1 = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim, dim),RReLU())     
        
        self.conv2 = GATConv(dim, dim, negative_slope=0.2, bias=True)
        
        self.lin_covert2 = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim, dim),RReLU())     
        
        self.conv3 = GATConv(dim, dim, negative_slope=0.2, bias=True)
        
        self.lin_covert3 = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                             RReLU(), Dropout(),Linear(dim*2, dim),RReLU())
        
        self.conv4 = GATConv(dim, dim, negative_slope=0.2, bias=True)
        
        self.lin_covert4 = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                             RReLU(), Dropout(),Linear(dim*2, dim),RReLU())
        
        self.lin_weight = Linear(8, dim*3, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*3)
        self.norm_x = BatchNorm1d(node_in)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        
        m = F.rrelu(self.conv1(out, data.edge_index))
        out = out + m
        out = self.lin_covert1(out)
        
        m = F.rrelu(self.conv2(out, data.edge_index))
        out = out + m
        out = self.lin_covert2(out)        
        
        m = F.rrelu(self.conv3(out, edge_index3))
        out = out + m
        out = self.lin_covert3(out)          

        m = F.rrelu(self.conv4(out, edge_index3))
        out = out + m
        out = self.lin_covert4(out)
            
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
        
        
class Net_int_2Edges_pool(torch.nn.Module):
    # use set2set for global pooling and cat results with local features
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges_pool, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        self.lin_edge_attr = torch.nn.Linear(edge_in, edge_dim)
        
        nn1 = Linear(edge_dim, dim * dim, bias=False)
        nn2 = Linear(edge_in3, dim * dim * 2 * 2, bias=False)
        
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = NNConv(dim*2, dim*2, nn2, aggr='mean', root_weight=False)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.lin_weight = Linear(8, dim*5*2, bias=False)
        self.lin_bias = Linear(8, 1, bias=False)
        self.norm = BatchNorm1d(dim*5*2)
        self.norm_x = BatchNorm1d(node_in)
        self.set2set = Set2Set(dim*2,processing_steps=3)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
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
        h = out.unsqueeze(0)
        for i in range(2):
            # using couping as edge
            m = F.rrelu(self.conv2(out, edge_index3, edge_attr3))
            out, h = self.gru2(m.unsqueeze(0), h)
            out = out.squeeze(0)  
            
        coupling_batch_index = data.batch[data.edge_index3[0]]
        pool = self.set2set(out, data.batch) # (m,d)
        pool = pool[coupling_batch_index] # (n_target,d)
        temp = out[data.edge_index3] # (2,n_target,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2,pool],1)  
        
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



class Net_int_2Edges_intHead(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges_intHead, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        self.lin_edge_attr = torch.nn.Linear(edge_in, edge_dim)
        
        nn1 = Linear(edge_dim, dim * dim, bias=False)
        nn2 = Linear(edge_in3, dim * dim * 2 * 2, bias=False)
        
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = NNConv(dim*2, dim*2, nn2, aggr='mean', root_weight=False)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.head = InteractionNet(edge_in3,[dim*3*2,dim*2,1],[F.relu,None])
        
        self.norm = BatchNorm1d(dim*3*2)
        self.norm_x = BatchNorm1d(node_in)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
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
        h = out.unsqueeze(0)
        for i in range(2):
            # using couping as edge
            m = F.rrelu(self.conv2(out, edge_index3, edge_attr3))
            out, h = self.gru2(m.unsqueeze(0), h)
            out = out.squeeze(0)     
            
        temp = out[data.edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)
        yhat = self.norm(yhat)
        yhat = self.head(data.edge_attr3,yhat)
        yhat = yhat.squeeze(1)
        
        if IsTrain:
            k = torch.sum(data.edge_attr3,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(data.y-yhat).unsqueeze(1)
            loss = torch.sum(torch.log(torch.sum(abs_ * data.edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex]))/nonzeroIndex.shape[0]
            return loss
        else:
            return yhat
        
class Net_int_2Edges_pool_intHead2(torch.nn.Module):
    # use pool in interaction head
    def __init__(self,dim=64,edge_dim=12,node_in=8,edge_in=19,edge_in3=8):
        super(Net_int_2Edges_pool_intHead2, self).__init__()
        self.lin_node = torch.nn.Linear(node_in, dim)
        self.lin_edge_attr = torch.nn.Linear(edge_in, edge_dim)
        
        nn1 = Linear(edge_dim, dim * dim, bias=False)
        nn2 = Linear(edge_in3, dim * dim * 2 * 2, bias=False)
        
        self.conv1 = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)
        self.gru1 = GRU(dim, dim)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2), \
                                     RReLU(), Dropout(),Linear(dim*2, dim*2),RReLU())
        
        self.conv2 = NNConv(dim*2, dim*2, nn2, aggr='mean', root_weight=False)
        self.gru2 = GRU(dim*2, dim*2)
        
        self.head = InteractionNet2(edge_in3+dim*4,[dim*3*2,dim,1],[F.relu,None])
        
        self.norm = BatchNorm1d(dim*3*2)
        self.norm_x = BatchNorm1d(node_in)
        self.set2set = Set2Set(dim*2,processing_steps=3)
        
    def forward(self, data,IsTrain=False):
        out = F.rrelu(self.lin_node(self.norm_x(data.x)))
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
        h = out.unsqueeze(0)
        for i in range(2):
            # using couping as edge
            m = F.rrelu(self.conv2(out, edge_index3, edge_attr3))
            out, h = self.gru2(m.unsqueeze(0), h)
            out = out.squeeze(0)  
            
        coupling_batch_index = data.batch[data.edge_index3[0]]
        pool = self.set2set(out, data.batch) # (m,d)
        pool = pool[coupling_batch_index] # (n_target,d)
        temp = out[data.edge_index3] # (2,n_target,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2],1)  
        
        yhat = self.norm(yhat)
        yhat = self.head(torch.cat([data.edge_attr3,pool],1),yhat)
        yhat = yhat.squeeze(1)
        
        if IsTrain:
            k = torch.sum(data.edge_attr3,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(data.y-yhat).unsqueeze(1)
            loss = torch.sum(torch.log(torch.sum(abs_ * data.edge_attr3[:,nonzeroIndex],0)/k[nonzeroIndex]))/nonzeroIndex.shape[0]
            return loss
        else:
            return yhat