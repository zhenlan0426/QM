#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:47:11 2019

@author: will
"""
from apex import amp
import math
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU,BatchNorm1d,Dropout,LeakyReLU
from torch_geometric.nn import NNConv#,GATConv
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.nn import init
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset,uniform
from torch_geometric.data import Data,DataLoader
from torch_geometric.utils import scatter_
from torch_scatter import scatter_mean,scatter_max
from torch_geometric.nn import MetaLayer

import time
import pickle
import numpy as np
import pandas as pd
import copy
import sys
import inspect

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Data ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''
data_dict = {'../Data/{}_data_ACSF.pickle':{'node_in':89,'edge_in':19,'edge_in4':1},\
             '../Data/{}_data_ACSF_expand.pickle':{'node_in':89,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_wACSF_expand_PCA.pickle':{'node_in':32,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_ACSF_expand_PCA.pickle':{'node_in':32,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_SOAP_expand_PCA.pickle':{'node_in':48,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_atomInfo.pickle':{'node_in':19,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_ACSF_SOAP_atomInfo.pickle':{'node_in':19+32+48,'edge_in':19+25,'edge_in4':1+25},\
             
             '../Data/{}_data_ACSF_expand_PCA_otherInfo.pickle':{'node_in':32,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_SOAP_expand_PCA_otherInfo.pickle':{'node_in':48,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_atomInfo_otherInfo.pickle':{'node_in':19,'edge_in':19+25,'edge_in4':1+25},\
             '../Data/{}_data_ACSF_SOAP_atomInfo_otherInfo.pickle':{'node_in':19+32+48,'edge_in':19+25,'edge_in4':1+25}}

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


special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec

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
        super(InteractionNet2, self).__init__()
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


class SimplyInteraction(torch.nn.Module):
    def __init__(self,xDim,factor=2,IntDim=8):
        # None in FunList mean identity func
        super(SimplyInteraction, self).__init__()
        self.w0 = nn.Parameter(torch.Tensor(IntDim,xDim,xDim*factor))
        self.b0 = nn.Parameter(torch.Tensor(IntDim,xDim*factor))
        self.w1 = nn.Parameter(torch.Tensor(IntDim,xDim*factor,1))
        self.b1 = nn.Parameter(torch.Tensor(IntDim,1))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.w0)
        init.kaiming_uniform_(self.w1)
        
        fan_in = self.w0.size(2)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b0, -bound, bound)
        fan_in = self.w1.size(2)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b1, -bound, bound)
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr3_old):
        out = F.relu(torch.einsum('np,dpq->ndq',edge_attr3,self.w0) + self.b0)
        out = torch.einsum('ndp,dpq->ndq',out,self.w1) + self.b1
        out = out.squeeze(2)
        return out[edge_attr3_old.to(torch.bool)]
    
def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan = init._calculate_correct_fan(tensor[0], mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
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

class feedforwardHead_Update(torch.nn.Module):
    def __init__(self,dim):
        factor = 2
        super(feedforwardHead_Update, self).__init__()
        self.linear = Sequential(Linear(dim, dim*factor),ReLU(), \
                                 Linear(dim*factor, 1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr3_old):
        yhat = self.linear(edge_attr3).squeeze(1)
        return yhat

class feedforwardCombineHead_Update(torch.nn.Module):
    def __init__(self,dim):
        factor = 2
        cat_factor = 4
        super(feedforwardCombineHead_Update, self).__init__()
        self.linear = Sequential(Dropout(0.33),Linear(dim*cat_factor, dim*cat_factor*factor),ReLU(), \
                                 Dropout(0.33),Linear(dim*cat_factor*factor, 1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr3_old):
        temp = x[edge_index3] # (2,N,d)
        yhat = torch.cat([temp.mean(0),temp[0]*temp[1],(temp[0]-temp[1])**2,edge_attr3],1)
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
    
class SimpleHead_type(torch.nn.Module):
    def __init__(self,dim,edge_in3,edge_in4):
        cat_factor = 2
        super(SimpleHead_type, self).__init__()
        self.linear = Sequential(#BatchNorm1d(dim*cat_factor),
                                 Linear(dim*cat_factor,dim*cat_factor*2),
                                 ReLU(inplace=True),
                                 #BatchNorm1d(dim*cat_factor*2),
                                 Linear(dim*cat_factor*2,1))
        
    def forward(self,x,edge_index3,edge_attr3,edge_attr4,batch):
        temp = x[edge_index3] # (2,N,d)
        yhat = torch.cat([temp[0],temp[1]],1)
        yhat = self.linear(yhat).squeeze(1)
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

class head_mol(torch.nn.Module):
    def __init__(self,dim,mol_shape):
        factor = 2
        super(head_mol, self).__init__()
        self.linear = Sequential(Linear(dim*2, dim*factor),ReLU(), \
                                 Linear(dim*factor, mol_shape))
        self.set2set = Set2Set(dim, processing_steps=3)
        
    def forward(self,out,batch):
        out = self.set2set(out, batch)
        out = self.linear(out).squeeze(1)
        return out
    
class head_mol2(torch.nn.Module):
    def __init__(self,dim,mol_shape):
        factor = 2
        super(head_mol2, self).__init__()
        self.linear = Sequential(Linear(dim, dim*factor),ReLU(), \
                                 Linear(dim*factor, mol_shape))
        
    def forward(self,u):
        out = self.linear(u).squeeze(1)
        return out
    
class head_atom(torch.nn.Module):
    def __init__(self,dim,atom_shape):
        factor = 2
        super(head_atom, self).__init__()
        self.linear = Sequential(Linear(dim, dim*factor),ReLU(), \
                                 Linear(dim*factor, atom_shape))
        
    def forward(self,out):
        out = self.linear(out).squeeze(1)
        return out

class head_edge(torch.nn.Module):
    def __init__(self,dim,edge_shape):
        factor = 2
        super(head_edge, self).__init__()
        self.linear = Sequential(Linear(dim, dim*factor),ReLU(), \
                                 Linear(dim*factor, edge_shape))
        
    def forward(self,out):
        out = self.linear(out).squeeze(1)
        return out
    
'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Conv ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''
class MessagePassing_edgeUpdate(torch.nn.Module):

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing_edgeUpdate, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(0)
                        if size[idx] != tmp.size(0):
                            raise ValueError(__size_error_msg__)

                        tmp = torch.index_select(tmp, 0, edge_index[idx])
                        message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out,edge_ = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        out = self.update(out, *update_args)

        return out,edge_


    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j


    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out

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
    def __init__(self,dim=64,edge_dim=12,aggr='mean'):
        super(schnet_block, self).__init__()
        
        nn = Sequential(BatchNorm1d(edge_dim),Linear(edge_dim, dim*2),ReLU(), \
                        BatchNorm1d(dim*2),Linear(dim*2, dim))
        self.conv = NNConv2(dim, dim, nn, aggr=aggr, root_weight=False)
        self.lin_covert = Sequential(BatchNorm1d(dim),Linear(dim, dim*2),ReLU(), \
                                     BatchNorm1d(dim*2),Linear(dim*2, dim))
        
    def forward(self, x, edge_index, edge_attr):
        m = F.relu(self.conv(x, edge_index, edge_attr))
        m = self.lin_covert(m)
        return x + m


class NNConv_block(torch.nn.Module):
    # use both types of edges
    def __init__(self,dim=64,edge_dim=12,aggr='mean'):
        super(NNConv_block, self).__init__()
        
        nn = Sequential(BatchNorm1d(edge_dim),Linear(edge_dim, dim*dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr, root_weight=False)
        self.gru = GRU(dim, dim)
        
    def forward(self, x, edge_index, edge_attr):
        m = F.relu(self.conv(x, edge_index, edge_attr))
        out, _ = self.gru(m.unsqueeze(0), x.unsqueeze(0))
        return out.squeeze(0)


class MEGNet(MessagePassing_edgeUpdate):
    def __init__(self,dim,aggr='mean'):
        super(MEGNet, self).__init__(aggr=aggr)
        cat_factor = 2
        multiple_factor = 3
        self.dim = dim
        self.v_update = Sequential(BatchNorm1d(dim*cat_factor),
                                    Linear(dim*cat_factor,dim*cat_factor*multiple_factor),
                                    LeakyReLU(inplace=True),
                                    BatchNorm1d(dim*cat_factor*multiple_factor),
                                    Linear(dim*cat_factor*multiple_factor,dim),
                                    LeakyReLU(inplace=True))
        
        self.e_update = Sequential(BatchNorm1d(dim*cat_factor),
                                    Linear(dim*cat_factor,dim*cat_factor*multiple_factor),
                                    LeakyReLU(inplace=True),
                                    BatchNorm1d(dim*cat_factor*multiple_factor),
                                    Linear(dim*cat_factor*multiple_factor,dim),
                                    LeakyReLU(inplace=True),)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        out = self.e_update(torch.cat([x_i+x_j,edge_attr],1))
        return out,out

    def update(self, aggr_out, x):
        return self.v_update(torch.cat([aggr_out,x],1))

    def __repr__(self):
        return 'MEGNet'

class MEGNet_block(torch.nn.Module):
    def __init__(self,dim,aggr='mean'):
        super(MEGNet_block, self).__init__()
        cat_factor = 1
        multiple_factor = 3        
        self.v_update =  Sequential(BatchNorm1d(dim*cat_factor),
                                    Linear(dim*cat_factor,dim*cat_factor*multiple_factor),
                                    LeakyReLU(inplace=True),
                                    BatchNorm1d(dim*cat_factor*multiple_factor),
                                    Linear(dim*cat_factor*multiple_factor,dim))
        self.e_update = Sequential( BatchNorm1d(dim*cat_factor),
                                    Linear(dim*cat_factor,dim*cat_factor*multiple_factor),
                                    BatchNorm1d(dim*cat_factor*multiple_factor),
                                    LeakyReLU(inplace=True),
                                    Linear(dim*cat_factor*multiple_factor,dim))        
        self.conv = MEGNet(dim,aggr=aggr)
    
    def forward(self, x, edge_index, edge_attr):
        x_new,edge_new = self.conv(x, edge_index, edge_attr)
        x_new = self.v_update(x_new)
        edge_new = self.e_update(edge_new)
        return x+x_new,edge_attr+edge_new
    
    def __repr__(self):
        return 'MEGNet_block'    
    
'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Main ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''


    

class GNN(torch.nn.Module):

    def __init__(self,reuse,block,head,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8,aggr='mean',interleave=False):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN, self).__init__()
        if interleave:
            assert layer1==layer2,'layer1 needs to be the same as layer2'
        self.interleave = interleave        
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),ReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),ReLU())
        
        if reuse:
            self.conv1 = nn.ModuleList([block(dim=dim,edge_dim=edge_in,aggr=aggr)] * layer1)
            self.conv2 = nn.ModuleList([block(dim=dim,edge_dim=edge_in3+edge_in4,aggr=aggr)] * layer2)
        else:
            self.conv1 = nn.ModuleList([block(dim=dim,edge_dim=edge_in,aggr=aggr) for _ in range(layer1)])
            self.conv2 = nn.ModuleList([block(dim=dim,edge_dim=edge_in3+edge_in4,aggr=aggr) for _ in range(layer2)])            
        
        self.head = head(dim,edge_in3,edge_in4)
        
    def forward(self, data,IsTrain=False,typeTrain=False):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        temp_ = torch.cat([data.edge_attr3,data.edge_attr4],1) 
        edge_attr3 = torch.cat([temp_,temp_],0)

        if self.interleave:
            for conv1,conv2 in zip(self.conv1,self.conv2):
                out = conv1(out,data.edge_index,data.edge_attr)
                out = conv2(out,edge_index3,edge_attr3)
        else:
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
            edge_attr3_old = data.edge_attr3[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3 = data.edge_attr3
            edge_attr4 = data.edge_attr4
            edge_attr3_old = data.edge_attr3
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr4,data.batch)
        
        if IsTrain:
            k = torch.sum(edge_attr3_old,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex])
            loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
            return loss,loss_perType
        else:
            return yhat
        
class GNN_edgeUpdate(torch.nn.Module):

    def __init__(self,reuse,block,head,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8,aggr='mean'):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN_edgeUpdate, self).__init__()
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge1 = Sequential(BatchNorm1d(edge_in),Linear(edge_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge2 = Sequential(BatchNorm1d(edge_in4+edge_in3),Linear(edge_in4+edge_in3, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())        
        if reuse:
            self.conv1 = block(dim=dim,aggr=aggr)
            self.conv2 = block(dim=dim,aggr=aggr)
        else:
            self.conv1 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer1)])
            self.conv2 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer2)])            
        
        self.head = head(dim)
        
    def forward(self, data,IsTrain=False,typeTrain=False,logLoss=True):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        n = data.edge_attr3.shape[0]
        temp_ = self.edge2(torch.cat([data.edge_attr3,data.edge_attr4],1))
        edge_attr3 = torch.cat([temp_,temp_],0)
        
        edge_attr = self.edge1(data.edge_attr)
        for conv in self.conv1:
            out,edge_attr = conv(out,data.edge_index,edge_attr)
        
        for conv in self.conv2:
            out,edge_attr3 = conv(out,edge_index3,edge_attr3)    
        
        edge_attr3 = edge_attr3[:n]
        if typeTrain:
            if IsTrain:
                y = data.y[data.type_attr]
            edge_attr3 = edge_attr3[data.type_attr]
            edge_index3 = data.edge_index3[:,data.type_attr]
            edge_attr3_old = data.edge_attr3[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3_old = data.edge_attr3
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr3_old)
        
        if IsTrain:
            k = torch.sum(edge_attr3_old,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            if logLoss:
                loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex])
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                return loss,loss_perType
            else:
                loss_perType[nonzeroIndex] = torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex]
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                loss_perType[nonzeroIndex] = torch.log(loss_perType[nonzeroIndex])
                return loss,loss_perType
        else:
            return yhat


class GNN_multiHead(torch.nn.Module):
    # for MEGNet only
    def __init__(self,reuse,block,head,head_mol,head_atom,head_edge,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8,mol_shape=4,atom_shape=10,edge_shape=4,aggr='mean'):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN_multiHead, self).__init__()
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge1 = Sequential(BatchNorm1d(edge_in),Linear(edge_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge2 = Sequential(BatchNorm1d(edge_in4+edge_in3),Linear(edge_in4+edge_in3, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())        
        if reuse:
            self.conv1 = block(dim=dim,aggr=aggr)
            self.conv2 = block(dim=dim,aggr=aggr)
        else:
            self.conv1 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer1)])
            self.conv2 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer2)])            
        
        self.head = head(dim)
        self.head_mol = head_mol(dim,mol_shape)
        self.head_atom = head_atom(dim,atom_shape)
        self.head_edge = head_edge(dim,edge_shape)
        
    def forward(self, data,IsTrain=False,typeTrain=False,logLoss=True,weight=None):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        n = data.edge_attr3.shape[0]
        temp_ = self.edge2(torch.cat([data.edge_attr3,data.edge_attr4],1))
        edge_attr3 = torch.cat([temp_,temp_],0)
        
        edge_attr = self.edge1(data.edge_attr)
        for conv in self.conv1:
            out,edge_attr = conv(out,data.edge_index,edge_attr)
        
        for conv in self.conv2:
            out,edge_attr3 = conv(out,edge_index3,edge_attr3)    
        
        edge_attr3 = edge_attr3[:n]
        if typeTrain:
            if IsTrain:
                y = data.y[data.type_attr]
            edge_attr3 = edge_attr3[data.type_attr]
            edge_index3 = data.edge_index3[:,data.type_attr]
            edge_attr3_old = data.edge_attr3[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3_old = data.edge_attr3
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr3_old)
        
        if IsTrain:
            if weight is None:
                loss_other = 0
            else:
                y_mol = self.head_mol(out,data.batch)
                y_atom = self.head_atom(out)
                y_edge = self.head_edge(edge_attr3)
                loss_other = weight * (torch.mean(torch.abs(data.y_mol - y_mol)) + \
                                       torch.mean(torch.abs(data.y_atom - y_atom)) + \
                                       torch.mean(torch.abs(data.y_coupling - y_edge)))

            k = torch.sum(edge_attr3_old,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            if logLoss:
                loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex])
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                return loss+loss_other,loss_perType         
            else:
                loss_perType[nonzeroIndex] = torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex]
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                loss_perType[nonzeroIndex] = torch.log(loss_perType[nonzeroIndex])
                return loss+loss_other,loss_perType
        else:
            return yhat


class GNN_multiHead_interleave(torch.nn.Module):
    # for MEGNet only
    def __init__(self,reuse,block,head,head_mol,head_atom,head_edge,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8,mol_shape=4,atom_shape=10,edge_shape=4,aggr='mean',interleave=False):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN_multiHead_interleave, self).__init__()
        if interleave:
            assert layer1==layer2,'layer1 needs to be the same as layer2'
        self.interleave = interleave
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge1 = Sequential(BatchNorm1d(edge_in),Linear(edge_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge2 = Sequential(BatchNorm1d(edge_in4+edge_in3),Linear(edge_in4+edge_in3, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())        
        if reuse:
            self.conv1 = block(dim=dim,aggr=aggr)
            self.conv2 = block(dim=dim,aggr=aggr)
        else:
            self.conv1 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer1)])
            self.conv2 = nn.ModuleList([block(dim=dim,aggr=aggr) for _ in range(layer2)])            
        
        self.head = head(dim)
        self.head_mol = head_mol(dim,mol_shape)
        self.head_atom = head_atom(dim,atom_shape)
        self.head_edge = head_edge(dim,edge_shape)
        
    def forward(self, data,IsTrain=False,typeTrain=False,logLoss=True,weight=None):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        n = data.edge_attr3.shape[0]
        temp_ = self.edge2(torch.cat([data.edge_attr3,data.edge_attr4],1))
        edge_attr3 = torch.cat([temp_,temp_],0)
        
        edge_attr = self.edge1(data.edge_attr)
        
        if self.interleave:
            for conv1,conv2 in zip(self.conv1,self.conv2):
                out,edge_attr = conv1(out,data.edge_index,edge_attr)
                out,edge_attr3 = conv2(out,edge_index3,edge_attr3)
        else:
            for conv in self.conv1:
                out,edge_attr = conv(out,data.edge_index,edge_attr)
            for conv in self.conv2:
                out,edge_attr3 = conv(out,edge_index3,edge_attr3)    
        
        edge_attr3 = edge_attr3[:n]
        if typeTrain:
            if IsTrain:
                y = data.y[data.type_attr]
            edge_attr3 = edge_attr3[data.type_attr]
            edge_index3 = data.edge_index3[:,data.type_attr]
            edge_attr3_old = data.edge_attr3[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3_old = data.edge_attr3
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr3_old)
        
        if IsTrain:
            if weight is None:
                loss_other = 0
            else:
                y_mol = self.head_mol(out,data.batch)
                y_atom = self.head_atom(out)
                y_edge = self.head_edge(edge_attr3)
                loss_other = weight * (torch.mean(torch.abs(data.y_mol - y_mol)) + \
                                       torch.mean(torch.abs(data.y_atom - y_atom)) + \
                                       torch.mean(torch.abs(data.y_coupling - y_edge)))

            k = torch.sum(edge_attr3_old,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            if logLoss:
                loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex])
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                return loss+loss_other,loss_perType         
            else:
                loss_perType[nonzeroIndex] = torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex]
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                loss_perType[nonzeroIndex] = torch.log(loss_perType[nonzeroIndex])
                return loss+loss_other,loss_perType
        else:
            return yhat
        
'''------------------------------------------------------------------------------------------------------------------'''
'''---------------------------------------------------- MetaLayer ----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''        

class UnitModel(torch.nn.Module):
    def __init__(self,dim,BatchNorm=True,factor=2,useMax=False):
        super(UnitModel, self).__init__()
        self.useMax = useMax
        if BatchNorm:
            self._mlp = Sequential(BatchNorm1d(dim*3),Linear(dim*3, dim*3*factor),LeakyReLU(), \
                                       BatchNorm1d(dim*3*factor),Linear(dim*3*factor, dim),LeakyReLU())
        else:
            self._mlp = Sequential(Linear(dim*3, dim*3*factor),LeakyReLU(), \
                                       Linear(dim*3*factor, dim),LeakyReLU())


class EdgeModel(UnitModel):
    def __init__(self,dim,BatchNorm=True,factor=2,useMax=False):
        super().__init__(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        if self.useMax:
            out = torch.cat([torch.max(src,dest), edge_attr, u[batch]], 1)
        else:
            out = torch.cat([src+dest, edge_attr, u[batch]], 1)
        return self._mlp(out)

class NodeModel(UnitModel):
    def __init__(self,dim,BatchNorm=True,factor=2,useMax=False):
        super().__init__(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if self.useMax:
            out,_ = scatter_max(edge_attr, edge_index[0], dim=0, dim_size=x.size(0))
        else:
            out = scatter_mean(edge_attr, edge_index[0], dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self._mlp(out)

class GlobalModel(UnitModel):
    def __init__(self,dim,BatchNorm=True,factor=2,useMax=False):
        super().__init__(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if self.useMax:
            out = torch.cat([u, scatter_max(x, batch, dim=0)[0],scatter_max(edge_attr, batch[edge_index[0]], dim=0)[0]], dim=1)
        else:
            out = torch.cat([u, scatter_mean(x, batch, dim=0),scatter_mean(edge_attr, batch[edge_index[0]], dim=0)], dim=1)
        return self._mlp(out)


class MetaLayer_block(torch.nn.Module):
    def __init__(self,dim,BatchNorm=True,factor=2,useMax=False):
        super(MetaLayer_block, self).__init__()
        if BatchNorm:
            self.v_update = Sequential(BatchNorm1d(dim),
                                        Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        BatchNorm1d(dim*factor),
                                        Linear(dim*factor,dim))
            self.e_update = Sequential(BatchNorm1d(dim),
                                        Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        BatchNorm1d(dim*factor),
                                        Linear(dim*factor,dim))
            self.u_update = Sequential(BatchNorm1d(dim),
                                        Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        BatchNorm1d(dim*factor),
                                        Linear(dim*factor,dim))
        else:
            self.v_update = Sequential(Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        Linear(dim*factor,dim))
            self.e_update = Sequential(Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        Linear(dim*factor,dim))
            self.u_update = Sequential(Linear(dim,dim*factor),
                                        LeakyReLU(inplace=True),
                                        Linear(dim*factor,dim))
            
        self.conv = MetaLayer(EdgeModel(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax), \
                              NodeModel(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax), \
                              GlobalModel(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax))
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        x_new, edge_attr_new, u_new = self.conv(x, edge_index, edge_attr, u, batch)
        x_new = self.v_update(x_new)
        edge_attr_new = self.e_update(edge_attr_new)
        u_new = self.u_update(u_new)
        return x+x_new,edge_attr+edge_attr_new,u+u_new
    
    def __repr__(self):
        return 'MetaLayer_block'   

class GNN_MataLayer(torch.nn.Module):
    # for MEGNet only
    def __init__(self,head,head_mol,head_atom,head_edge,dim,layer1,layer2,factor,\
                 node_in,edge_in,edge_in4,edge_in3=8,mol_shape=4,atom_shape=10,edge_shape=4,BatchNorm=True,useMax=False,interleave=False):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(GNN_MataLayer, self).__init__()
        self.useMax = useMax
        if interleave:
            assert layer1==layer2,'layer1 needs to be the same as layer2'
        self.interleave = interleave
        self.lin_node = Sequential(BatchNorm1d(node_in),Linear(node_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge1 = Sequential(BatchNorm1d(edge_in),Linear(edge_in, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())

        self.edge2 = Sequential(BatchNorm1d(edge_in4+edge_in3),Linear(edge_in4+edge_in3, dim*factor),LeakyReLU(), \
                                   BatchNorm1d(dim*factor),Linear(dim*factor, dim),LeakyReLU())        
        self.u_mlp = Sequential(BatchNorm1d(2*dim),Linear(2*dim, 2*dim*factor),LeakyReLU(), \
                                   BatchNorm1d(2*dim*factor),Linear(2*dim*factor, dim),LeakyReLU())        
        
        self.conv1 = nn.ModuleList([MetaLayer_block(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax) for _ in range(layer1)])
        self.conv2 = nn.ModuleList([MetaLayer_block(dim,BatchNorm=BatchNorm,factor=factor,useMax=useMax) for _ in range(layer2)])            
        
        self.head = head(dim)
        self.head_mol = head_mol(dim,mol_shape)
        self.head_atom = head_atom(dim,atom_shape)
        self.head_edge = head_edge(dim,edge_shape)
        
    def forward(self, data,IsTrain=False,typeTrain=False,logLoss=True,weight=None):
        out = self.lin_node(data.x)
        # edge_*3 only does not repeat for undirected graph. Hence need to add (j,i) to (i,j) in edges
        edge_index3 = torch.cat([data.edge_index3,data.edge_index3[[1,0]]],1)
        n = data.edge_attr3.shape[0]
        temp_ = self.edge2(torch.cat([data.edge_attr3,data.edge_attr4],1))
        edge_attr3 = torch.cat([temp_,temp_],0)
        edge_attr = self.edge1(data.edge_attr)
        
        if self.useMax:
            u = torch.cat([scatter_max(out, data.batch, dim=0)[0],scatter_max(edge_attr, data.batch[data.edge_index[0]], dim=0)[0]], dim=1)
        else:
            u = torch.cat([scatter_mean(out, data.batch, dim=0),scatter_mean(edge_attr, data.batch[data.edge_index[0]], dim=0)], dim=1)
        u = self.u_mlp(u)
        
        if self.interleave:
            for conv1,conv2 in zip(self.conv1,self.conv2):
                out,edge_attr,u = conv1(out, data.edge_index, edge_attr, u, data.batch)
                out,edge_attr3,u = conv2(out,edge_index3,edge_attr3, u, data.batch)
        else:
            for conv in self.conv1:
                out,edge_attr,u = conv(out, data.edge_index, edge_attr, u, data.batch)
            for conv in self.conv2:
                out,edge_attr3,u = conv(out,edge_index3,edge_attr3, u, data.batch) 
        
        edge_attr3 = edge_attr3[:n]
        if typeTrain:
            if IsTrain:
                y = data.y[data.type_attr]
            edge_attr3 = edge_attr3[data.type_attr]
            edge_index3 = data.edge_index3[:,data.type_attr]
            edge_attr3_old = data.edge_attr3[data.type_attr]
        else:
            if IsTrain:
                y = data.y
            edge_index3 = data.edge_index3
            edge_attr3_old = data.edge_attr3
            
        yhat = self.head(out,edge_index3,edge_attr3,edge_attr3_old)
        
        if IsTrain:
            if weight is None:
                loss_other = 0
            else:
                y_mol = self.head_mol(u)
                y_atom = self.head_atom(out)
                y_edge = self.head_edge(edge_attr3)
                loss_other = weight * (torch.mean(torch.abs(data.y_mol - y_mol)) + \
                                       torch.mean(torch.abs(data.y_atom - y_atom)) + \
                                       torch.mean(torch.abs(data.y_coupling - y_edge)))

            k = torch.sum(edge_attr3_old,0)
            nonzeroIndex = torch.nonzero(k).squeeze(1)
            abs_ = torch.abs(y-yhat).unsqueeze(1)
            loss_perType = torch.zeros(8,device='cuda:0')
            if logLoss:
                loss_perType[nonzeroIndex] = torch.log(torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex])
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                return loss+loss_other,loss_perType         
            else:
                loss_perType[nonzeroIndex] = torch.sum(abs_ * edge_attr3_old[:,nonzeroIndex],0)/k[nonzeroIndex]
                loss = torch.sum(loss_perType)/nonzeroIndex.shape[0]
                loss_perType[nonzeroIndex] = torch.log(loss_perType[nonzeroIndex])
                return loss+loss_other,loss_perType
        else:
            return yhat

'''------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------- utility -----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''


#class Data2(Data):
#    def apply_ignore_index(self, func, *keys):
#        r"""Applies the function :obj:`func` to all tensor attributes
#        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
#        all present attributes.
#        """
#        for key, item in self(*keys):
#            if torch.is_tensor(item):
#                if 'index' not in key:
#                    self[key] = func(item)
#        return self
#    
#    def to(self, device, *keys):
#        r"""Performs tensor dtype and/or device conversion to all attributes
#        :obj:`*keys`.
#        If :obj:`*keys` is not given, the conversion is applied to all present
#        attributes."""
#        if 'cuda' in str(device):
#            return self.apply(lambda x: x.to(device), *keys)
#        else:
#            return self.apply_ignore_index(lambda x: x.to(device), *keys)

#def get_data2(data,batch_size):
#    with open(data.format('train'), 'rb') as handle:
#        train_data = pickle.load(handle)
#    with open(data.format('val'), 'rb') as handle:
#        val_data = pickle.load(handle)
#    
#    train_list = [Data2(**d) for d in train_data]
#    train_dl = DataLoader(train_list,batch_size,shuffle=True)
#    val_list = [Data2(**d) for d in val_data]
#    val_dl = DataLoader(val_list,batch_size,shuffle=False)
#    
#    return train_dl,val_dl


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

def train(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,train_loss_list=None,val_loss_list=None,scheduler=None):
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
        val_loss = val_loss/j
        if val_loss<lossBest:
            lossBest = val_loss
            bestWeight = copy.deepcopy(model.state_dict())
            
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType/j])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType/j)
        if scheduler is not None:
            scheduler.step(val_loss)
        
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    
    # load best model
    model.load_state_dict(bestWeight)
    return model,train_loss_list,val_loss_list

def train_earlyStop(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,train_loss_list=None,val_loss_list=None,scheduler=None,threshold=0):
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
        
        
        val_loss = val_loss/j
        # early stop
        if (epoch==15) and (val_loss>threshold):
            print('\nEarly Stop\n')
            return None,None,None
        
        # save model
        if val_loss<lossBest:
            lossBest = val_loss
            torch.save({'model_state_dict': model.state_dict()},'../Model/tmp.tar')
            
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType/j])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType/j)
        if scheduler is not None:
            scheduler.step(val_loss)
        
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    
    # load best model
    checkpoint = torch.load('../Model/tmp.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model,train_loss_list,val_loss_list

def train_type(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,train_loss_list=None,val_loss_list=None,scheduler=None,logLoss=True,UseAmp=False,AMP_clip=False):
    # for MEGNet
    since = time.time()
    
    lossBest = [1e6] * 8
    bestWeight = [None] * 8
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
            loss,loss_perType = model(data,True,typeTrain,logLoss)
            
            if UseAmp:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if AMP_clip:
                clip_grad_value_(amp.master_params(opt), clip)
            else:
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
        val_loss_perType = val_loss_perType/j
        for index_ in range(8):
            if val_loss_perType[index_]<lossBest[index_]:
                lossBest[index_] = val_loss_perType[index_]
                bestWeight[index_] = copy.deepcopy(model.state_dict())
            
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss/j,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType)
        if scheduler is not None:
            scheduler.step(val_loss/j)
            
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    
    return model,train_loss_list,val_loss_list,bestWeight


def train_type_earlyStop(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,\
                         train_loss_list=None,val_loss_list=None,scheduler=None,logLoss=True,weight=None,threshold=0):
    # add early stop and weight for hyper Search
    # add early stop if nan for hyper3
    since = time.time()
    
    lossBest = [1e6] * 8
    bestWeight = [None] * 8
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
            loss,loss_perType = model(data,True,typeTrain,logLoss,weight)
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
                loss,loss_perType = model(data,True,typeTrain,True,None)
                val_loss += loss.item()
                val_loss_perType += loss_perType.cpu().detach().numpy()
        
        # early stop
        val_loss = val_loss/j
        if (epoch==20) and (val_loss>threshold):
            print('-----stop due to poor performance-----')
            return None,None,None,None
        
        # check nan
        if np.any(np.isnan(val_loss_perType)):
            print('-----stop due to nan-----')
            return None,None,None,None
        
        # save model
        val_loss_perType = val_loss_perType/j
        for index_ in range(8):
            if val_loss_perType[index_]<lossBest[index_]:
                lossBest[index_] = val_loss_perType[index_]
                bestWeight[index_] = copy.deepcopy(model.state_dict())
            
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType)
        if scheduler is not None:
            scheduler.step(val_loss)
            
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    return model,train_loss_list,val_loss_list,bestWeight

def train_type_earlyStop_5fold(opt,model,epochs,train_dl,val_dl,paras,clip,typeTrain=False,\
                         train_loss_list=None,val_loss_list=None,scheduler=None,logLoss=True,weight=None,patience=6):
    # add early stop for 5 fold
    since = time.time()
    counter = 0 
    lossBest = np.ones(8)*1e6
    bestWeight = [None] * 8
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
            loss,loss_perType = model(data,True,typeTrain,logLoss,weight)
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
                loss,loss_perType = model(data,True,typeTrain,True,None)
                val_loss += loss.item()
                val_loss_perType += loss_perType.cpu().detach().numpy()
        
        # save model
        val_loss_perType = val_loss_perType/j
        for index_ in range(8):
            if val_loss_perType[index_]<lossBest[index_]:
                lossBest[index_] = val_loss_perType[index_]
                bestWeight[index_] = copy.deepcopy(model.state_dict())
                
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch+epoch0,train_loss/i,val_loss,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType])))
        train_loss_list.append(train_loss_perType/i)
        val_loss_list.append(val_loss_perType)
        if scheduler is not None:
            scheduler.step(val_loss)
                
        # early stop
        if np.any(val_loss_perType<lossBest):
            counter = 0
        else:
            counter+= 1
            if counter >= patience:
                print('----early stop at epoch {}----'.format(epoch))
                return model,train_loss_list,val_loss_list,bestWeight
            
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    return model,train_loss_list,val_loss_list,bestWeight
    
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

def save_results2(train_loss_perType,val_loss_perType,*nameList):
    epochs = len(train_loss_perType)
    results = pd.DataFrame([nameList for _ in range(epochs)],dtype=str)
    temp = pd.DataFrame(np.concatenate([np.arange(epochs,dtype=np.int)[:,np.newaxis],np.stack(train_loss_perType,0),np.stack(val_loss_perType,0)],1),
             columns=['epochs']+['train_type_{}'.format(i) for i in range(8)] + ['val_type_{}'.format(i) for i in range(8)])
    results = pd.concat([results,temp],1)
    results.to_csv('../Data/results_{}.csv'.\
               format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                for i in nameList])),index=False)    
    
def save_model(model,opt,reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr='base'):
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epochs':epochs
            }, '../Model/{}.tar'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                    for i in [reuse,block,head,data,batch_size,dim,clip,\
                                          layer1,layer2,factor,epochs,postStr]])))


def save_model_type(bestWeight,opt,reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr='base'):
    opt_state = opt.state_dict()
    for i,w in enumerate(bestWeight):
        torch.save({'model_state_dict': w,
                'optimizer_state_dict': opt_state,
                'epochs':epochs
                }, '../Model/{}.tar'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                        for i in [reuse,block,head,data,batch_size,dim,clip,\
                                              layer1,layer2,factor,epochs,'type_'+str(i)+postStr]])))    

def save_model_type2(bestWeight,opt,*nameList):
    opt_state = opt.state_dict()
    for j,w in enumerate(bestWeight):
        torch.save({'model_state_dict': w,
                'optimizer_state_dict': opt_state,
                }, '../Model/{}.tar'.format('_'.join([str(i).split('}')[1] if '}' in str(i) else str(i) \
                                        for i in list(nameList)+['type_'+str(j)] ])))  
    
def make_submission(reuse,block,head,data,batch_size,dim,clip,layer1,layer2,factor,epochs,postStr='base'):
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
    
def average_submission(lol,submission,name='combine_type.csv'):
    # take in a 8 element list of list, [[sub1,sub2],...,[]]
    # i-th element contains a list of string for best submission for type-i, that will be averaged over
    for i,type_i in enumerate(lol):
        for sub in type_i:
            submit_df = pd.read_csv('../Submission/'+sub)['type_'+str(i)]
            submission = pd.concat([submission,submit_df],1)
    
    temp = submission.iloc[:,2:].mean(1)
    submission = submission[['id','scalar_coupling_constant']]
    submission['scalar_coupling_constant'] = temp
    submission.to_csv('../Submission/'+name,index=False)      