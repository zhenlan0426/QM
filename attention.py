#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:00:54 2019

@author: will
"""

from torch.utils.data import Dataset
import torch
from torch.nn import Sequential,Linear,BatchNorm1d,Dropout,MultiheadAttention,LayerNorm,TransformerEncoderLayer
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import clip_grad_value_
from torch.nn.init import xavier_uniform_

import math
import time
import numpy as np
import copy

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Data ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

class attentionDataset(Dataset):
    def __init__(self, node, edge, y=None):
        self.node = node # a list
        self.edge = edge
        self.y = y

    def __len__(self):
        return len(self.node)

    def __getitem__(self, idx):
        if self.y is None:
            return self.node[idx],self.edge[idx]
        else:
            return self.node[idx],self.edge[idx],self.y[idx]
        
def collate_fn(batch):
    if len(batch[0]) == 3:
        node,edge,y = zip(*batch)
    else:
        node,edge = zip(*batch)
    out = torch.nn.utils.rnn.pad_sequence(node)
    mask = (out==0).all(2).T
    edge = torch.stack(edge)[None,:,:]
    y = torch.stack(y)
    return out,mask,edge,y    


'''------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------- Model ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

class SimplyHead(torch.nn.Module):
    def __init__(self,dim,factor=2):
        # None in FunList mean identity func
        super(SimplyHead, self).__init__()
        self.linear = Sequential(BatchNorm1d(dim+8),
                                 Linear(dim+8,dim*factor),
                                 nn.LeakyReLU(inplace=True),
                                 BatchNorm1d(dim*factor),
                                 Linear(dim*factor,1))
        
    def forward(self,edge_attr3,edge_attr3_old):
        out = self.linear(torch.cat([edge_attr3,edge_attr3_old],1))
        return out.squeeze(1)

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
        
    def forward(self,edge_attr3,edge_attr3_old):
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

class TransformerDecoderLayer(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)


        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
       
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
class Attention(torch.nn.Module):
    # for MEGNet only
    def __init__(self,dim,encoder_layer,decoder_layer,head_d,head,node_d=8,edge_d=9,dropout=0.1,dim_feedforward=1024):
        # block,head are nn.Module
        # node_in,edge_in are dim for bonding and edge_in4,edge_in3 for coupling
        super(Attention, self).__init__()
        self.lin_node = Sequential(LayerNorm(node_d),Linear(node_d, dim))
        self.lin_edge = Sequential(LayerNorm(edge_d),Linear(edge_d, dim))
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(encoder_layer)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(decoder_layer)])
        self.norm = BatchNorm1d(dim)
        self.head = head(dim)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, out,mask,edge,y=None,logLoss=True):
        out = self.lin_node(out)
        edge2 = self.lin_edge(edge)

        for f in self.encoder_layers:
            out = f(out,src_key_padding_mask=mask)
        
        for f in self.decoder_layers:
            edge2 = f(edge2, out,memory_key_padding_mask=mask)
            
        edge2 = self.norm(edge2.squeeze(0))
        #edge2 = edge2.squeeze(0)
        edge_attr3_old = edge.squeeze(0)[:,:8]
        yhat = self.head(edge2,edge_attr3_old)
        
        if y is None:
            return yhat
        else:
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


'''------------------------------------------------------------------------------------------------------------------'''
'''---------------------------------------------------- Utility -----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''
            
def train_type(opt,model,epochs,train_dl,val_dl,paras,clip,\
               scheduler=None,logLoss=True,weight=None,patience=6,saveModelEpoch=50):
    # add early stop for 5 fold
    since = time.time()
    counter = 0 
    lossBest = np.ones(8)*1e6
    bestWeight = [None] * 8
    bestOpt = [None] * 8
        
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        np.random.seed()
        train_loss = 0
        train_loss_perType = np.zeros(8)
        val_loss = 0
        val_loss_perType = np.zeros(8)
        
        for i,(out,mask,edge,y) in enumerate(train_dl):
            out,mask,edge,y = out.to('cuda:0'),mask.to('cuda:0'),edge.to('cuda:0'),y.to('cuda:0')
            loss,loss_perType = model(out,mask,edge,y,logLoss)
            loss.backward()
            clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            train_loss_perType += loss_perType.cpu().detach().numpy()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,(out,mask,edge,y) in enumerate(val_dl):
                out,mask,edge,y = out.to('cuda:0'),mask.to('cuda:0'),edge.to('cuda:0'),y.to('cuda:0')
                loss,loss_perType = model(out,mask,edge,y,True)
                val_loss += loss.item()
                val_loss_perType += loss_perType.cpu().detach().numpy()
        val_loss = val_loss/j
        
        # save model
        val_loss_perType = val_loss_perType/j
        for index_ in range(8):
            if val_loss_perType[index_]<lossBest[index_]:
                lossBest[index_] = val_loss_perType[index_]
                if epoch>saveModelEpoch:
                    bestWeight[index_] = copy.deepcopy(model.state_dict())
                    bestOpt[index_] = copy.deepcopy(opt.state_dict())
                    
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}, \ntrain_vector: {}, \nval_vector  : {}\n'.format(epoch,train_loss/i,val_loss,\
                                                            '|'.join(['%+.2f'%i for i in train_loss_perType/i]),\
                                                            '|'.join(['%+.2f'%i for i in val_loss_perType])))
        if scheduler is not None:
            scheduler.step(val_loss)
                
        # early stop
        if np.any(val_loss_perType==lossBest):
            counter = 0
        else:
            counter+= 1
            if counter >= patience:
                print('----early stop at epoch {}----'.format(epoch))
                return model,bestOpt,bestWeight
            
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    return model,bestOpt,bestWeight            