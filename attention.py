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
from apex import amp
#from torch.nn.init import xavier_uniform_

import math
import time
import numpy as np
import copy

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Data ------------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

#class attentionDataset(Dataset):
#    def __init__(self, node, edge, y=None):
#        self.node = node # a list
#        self.edge = edge
#        self.y = y
#
#    def __len__(self):
#        return len(self.node)
#
#    def __getitem__(self, idx):
#        if self.y is None:
#            return self.node[idx],self.edge[idx]
#        else:
#            return self.node[idx],self.edge[idx],self.y[idx]
#        
#def collate_fn(batch):
#    if len(batch[0]) == 3:
#        node,edge,y = zip(*batch)
#    else:
#        node,edge = zip(*batch)
#    out = torch.nn.utils.rnn.pad_sequence(node)
#    mask = (out==0).all(2).T
#    edge = torch.stack(edge)[None,:,:]
#    y = torch.stack(y)
#    return out,mask,edge,y    

class attentionDataset(Dataset):
    def __init__(self, node, edge, ind, y=None):
        self.node = node # a list
        self.edge = edge
        self.ind = ind
        self.y = y

    def __len__(self):
        return len(self.node)

    def __getitem__(self, idx):
        if self.y is None:
            return self.node[idx],self.edge[idx],self.ind[idx]
        else:
            return self.node[idx],self.edge[idx],self.ind[idx],self.y[idx]
        
def collate_fn(batch):
    batch_len = len(batch[0])
    if batch_len == 4:
        node,edge,ind,y = zip(*batch)
    else:
        node,edge,ind = zip(*batch)
    out = torch.nn.utils.rnn.pad_sequence(node)
    mask = (out==0).all(2).T
    edge = torch.stack(edge)
    
    ind = torch.stack(ind)
    t,n,_ = out.shape
    m1 = torch.zeros((n,t))
    m1[range(n),ind[:,0]] = 1
    m1[range(n),ind[:,1]] = 1
    m1 = m1.T[...,None]
    
    m2 = torch.zeros((n,29)) # 29 is max number of atoms
    m2[range(n),ind[:,0]] = 1
    m2[range(n),ind[:,1]] = 1
    if batch_len == 4:
        return torch.cat([out,m1],2),mask,torch.cat([edge,m2],1)[None,:,:],torch.stack(y) 
    else:
        return torch.cat([out,m1],2),mask,torch.cat([edge,m2],1)[None,:,:]

def collate_fn2(batch):
    # return ind for indexing later
    batch_len = len(batch[0])
    if batch_len == 4:
        node,edge,ind,y = zip(*batch)
    else:
        node,edge,ind = zip(*batch)
    out = torch.nn.utils.rnn.pad_sequence(node)
    mask = (out==0).all(2).T
    edge = torch.stack(edge)
    
    ind = torch.stack(ind)
    t,n,_ = out.shape
    m1 = torch.zeros((n,t))
    m1[range(n),ind[:,0]] = 1
    m1[range(n),ind[:,1]] = 1
    m1 = m1.T[...,None]
    
    if batch_len == 4:
        return torch.cat([out,m1],2),mask,edge[None,:,:-1],ind,torch.stack(y) 
    else:
        return torch.cat([out,m1],2),mask,edge[None,:,:-1],ind
    
def collate_fn3(batch):
    # return ind[:,1],edge[None,:,:] for xyz
    batch_len = len(batch[0])
    if batch_len == 4:
        node,edge,ind,y = zip(*batch)
    else:
        node,edge,ind = zip(*batch)
    out = torch.nn.utils.rnn.pad_sequence(node)
    mask = (out==0).all(2).T
    edge = torch.stack(edge)
    
    ind = torch.stack(ind)
    t,n,_ = out.shape
    m1 = torch.zeros((n,t))
    m1[range(n),ind[:,0]] = 1
    m1[range(n),ind[:,1]] = 1
    m1 = m1.T[...,None]
    
    if batch_len == 4:
        return torch.cat([out,m1],2),mask,edge[None,:,:],ind[:,1],torch.stack(y) 
    else:
        return torch.cat([out,m1],2),mask,edge[None,:,:],ind[:,1]


class attentionDataset2(Dataset):
    # for mol-level attention
    def __init__(self, atom_list, coupling_list, index_list, target_list=None):
        self.atom_list = atom_list # a list
        self.coupling_list = coupling_list
        self.index_list = index_list
        self.target_list = target_list

    def __len__(self):
        return len(self.atom_list)

    def __getitem__(self, idx):
        if self.target_list is None:
            return self.atom_list[idx],self.coupling_list[idx],self.index_list[idx]
        else:
            return self.atom_list[idx],self.coupling_list[idx],self.index_list[idx],self.target_list[idx]
        
def collate_fn4(batch):
    # for mol-level attention
    batch_len = len(batch[0])
    if batch_len == 4:
        atom_list,coupling_list,index_list,target_list = zip(*batch)
    else:
        atom_list,coupling_list,index_list = zip(*batch)
        
    atom = torch.nn.utils.rnn.pad_sequence(atom_list)
    coupling = torch.nn.utils.rnn.pad_sequence(coupling_list)
    index_ = torch.nn.utils.rnn.pad_sequence(index_list)
    atom_mask = (atom==0).all(2).T
    coupling_mask = (coupling==0).all(2).T

    if batch_len == 4:
        return atom,atom_mask,coupling,coupling_mask,index_,torch.cat(target_list) 
    else:
        return atom,atom_mask,coupling,coupling_mask,index_
    
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

class TransformerEncoderLayer_BN(torch.nn.Module):
    # use BatchNorm instead of LayerNorm
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoderLayer_BN, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model)
        self.norm2 = BatchNorm1d(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src.transpose(1,2)).transpose(1,2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src.transpose(1,2)).transpose(1,2)
        return src
    
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


class TransformerDecoderLayer_BN(torch.nn.Module):
    # use BatchNorm instead of LayerNorm
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransformerDecoderLayer_BN, self).__init__()

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm2 = BatchNorm1d(d_model)
        self.norm3 = BatchNorm1d(d_model)

        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
       
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt.transpose(1,2)).transpose(1,2)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt.transpose(1,2)).transpose(1,2)
        return tgt
    
class Attention(torch.nn.Module):
    def __init__(self,dim,encoder_layer,decoder_layer,head_d,head,EncoderLayer,DecoderLayer,\
                 node_d=8+1,edge_d=9+29,dropout=0.1,dim_feedforward=1024):
        super(Attention, self).__init__()
#        self.lin_node = Sequential(LayerNorm(node_d),Linear(node_d, dim))
#        self.lin_edge = Sequential(LayerNorm(edge_d),Linear(edge_d, dim))
        self.lin_node = Linear(node_d, dim)
        self.lin_edge = Linear(edge_d, dim)
        self.norm_node = BatchNorm1d(node_d)
        self.norm_edge = BatchNorm1d(edge_d)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(encoder_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(decoder_layer)])
        self.norm = BatchNorm1d(dim)
        self.head = head(dim)
#        self._reset_parameters()
#
#    def _reset_parameters(self):
#        r"""Initiate parameters in the transformer model."""
#        for p in self.parameters():
#            if p.dim() > 1:
#                xavier_uniform_(p)

    def forward(self, out,mask,edge,y=None,logLoss=True):
        out = self.lin_node(self.norm_node(out.transpose(1,2)).transpose(1,2))
        edge2 = self.lin_edge(self.norm_edge(edge.transpose(1,2)).transpose(1,2))

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
            
            
class Attention2(torch.nn.Module):
    # no decoder.
    def __init__(self,dim,encoder_layer,head_d,head,EncoderLayer,\
                 node_d=8,edge_d=9,dropout=0.1,dim_feedforward=1024):
        super(Attention2, self).__init__()
        self.lin_node = Linear(node_d+edge_d, dim)
        self.norm_node = BatchNorm1d(node_d+edge_d)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(encoder_layer)])
        self.norm = BatchNorm1d(dim)
        self.head = head(dim)

    def forward(self, out,mask,edge,y=None,logLoss=True):
        edge_bc = torch.repeat_interleave(edge,out.shape[0],0)
        out = torch.cat([out,edge_bc],2)
        out = self.lin_node(self.norm_node(out.transpose(1,2)).transpose(1,2))

        for f in self.encoder_layers:
            out = f(out,src_key_padding_mask=mask)
            
        edge2 = self.norm(out.max(0)[0])
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

class Attention3(torch.nn.Module):
    # no decoder. + indexing instead of max
    def __init__(self,dim,encoder_layer,head_d,head,EncoderLayer,\
                 node_d=8+1,edge_d=8,dropout=0.1,dim_feedforward=1024,catFactor=2):
        super(Attention3, self).__init__()
        self.lin_node = Linear(node_d+edge_d, dim)
        self.norm_node = BatchNorm1d(node_d+edge_d)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(encoder_layer)])
        self.head = head(dim*catFactor)

    def forward(self, out,mask,edge,ind,y=None,logLoss=True):
        edge_bc = torch.repeat_interleave(edge,out.shape[0],0)
        n = out.shape[1]
        out = torch.cat([out,edge_bc],2)
        out = self.lin_node(self.norm_node(out.transpose(1,2)).transpose(1,2))

        for f in self.encoder_layers:
            out = f(out,src_key_padding_mask=mask)
        
        atom0 = out[ind[:,0],range(n)]
        atom1 = out[ind[:,1],range(n)]
        edge2 = torch.cat([atom0,atom1],1)
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

class Attention_mol(torch.nn.Module):
    def __init__(self,dim,encoder_layer,decoder_layer,head_d,head,EncoderLayer,DecoderLayer,\
                 node_d=8,edge_d=15,dropout=0.1,dim_feedforward=1024,catFactor=2):
        super(Attention_mol, self).__init__()
#        self.lin_node = Sequential(LayerNorm(node_d),Linear(node_d, dim))
#        self.lin_edge = Sequential(LayerNorm(edge_d),Linear(edge_d, dim))
        self.lin_node = Linear(node_d, dim)
        self.lin_edge = Linear(edge_d+dim*catFactor, dim)
        self.norm_node = BatchNorm1d(node_d)
        self.norm_edge = BatchNorm1d(edge_d)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(encoder_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim,head_d,dim_feedforward=dim_feedforward,dropout=dropout) for _ in range(decoder_layer)])
        self.norm = BatchNorm1d(dim)
        self.head = head(dim)
        self.dim = dim
#        self._reset_parameters()
#
#    def _reset_parameters(self):
#        r"""Initiate parameters in the transformer model."""
#        for p in self.parameters():
#            if p.dim() > 1:
#                xavier_uniform_(p)

    def forward(self, out,src_mask,edge,edge_mask,ind,y=None,logLoss=True):
        # ind has shape (T', N, 2)
        out = self.lin_node(self.norm_node(out.transpose(1,2)).transpose(1,2))
        
        for f in self.encoder_layers:
            out = f(out,src_key_padding_mask=src_mask)
        
        atom0 = torch.gather(out,0,ind[:,:,0].unsqueeze(-1).expand(-1,-1,self.dim))
        atom1 = torch.gather(out,0,ind[:,:,1].unsqueeze(-1).expand(-1,-1,self.dim))
        edge2 = self.lin_edge(torch.cat([atom0,atom1,self.norm_edge(edge.transpose(1,2)).transpose(1,2)],2))
        
        for f in self.decoder_layers:
            edge2 = f(edge2, out,memory_key_padding_mask=src_mask,tgt_key_padding_mask=edge_mask)
            
        edge2 = self.norm(edge2.transpose(1,2)).transpose(1,2)
        #edge2 = edge2.squeeze(0)
        edge_attr3_old = edge.transpose(0,1)[~edge_mask][:,:8]
        yhat = self.head(edge2,edge_attr3_old,edge_mask)
        
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


class SimplyInteraction_mol(torch.nn.Module):
    def __init__(self,xDim,factor=2,IntDim=8):
        # None in FunList mean identity func
        super(SimplyInteraction_mol, self).__init__()
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
        
    def forward(self,edge_attr3,edge_attr3_old,edge_mask):
        edge_attr3 = edge_attr3.transpose(0,1)[~edge_mask]
        out = F.relu(torch.einsum('np,dpq->ndq',edge_attr3,self.w0) + self.b0)
        out = torch.einsum('ndp,dpq->ndq',out,self.w1) + self.b1
        out = out.squeeze(2)
        return out[edge_attr3_old.to(torch.bool)]
    

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


def train_type2(opt,model,epochs,train_dl,val_dl,paras,clip,\
               scheduler=None,logLoss=True,weight=None,patience=6,saveModelEpoch=50):
    # add ind
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
        
        for i,(out,mask,edge,ind,y) in enumerate(train_dl):
            out,mask,edge,ind,y = out.to('cuda:0'),mask.to('cuda:0'),edge.to('cuda:0'),ind.to('cuda:0'),y.to('cuda:0')
            loss,loss_perType = model(out,mask,edge,ind,y,logLoss)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            clip_grad_value_(amp.master_params(opt),clip)
            #clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            train_loss_perType += loss_perType.cpu().detach().numpy()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,(out,mask,edge,ind,y) in enumerate(val_dl):
                out,mask,edge,ind,y = out.to('cuda:0'),mask.to('cuda:0'),edge.to('cuda:0'),ind.to('cuda:0'),y.to('cuda:0')
                loss,loss_perType = model(out,mask,edge,ind,y,True)
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

def train_type_mol(opt,model,epochs,train_dl,val_dl,paras,clip,\
               scheduler=None,logLoss=True,weight=None,patience=6,saveModelEpoch=50):
    # add ind
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
        
        for i,data in enumerate(train_dl):
            data = [d.to('cuda:0') for d in data]
            loss,loss_perType = model(*data,logLoss=logLoss)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            clip_grad_value_(amp.master_params(opt),clip)
            #clip_grad_value_(paras,clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
            train_loss_perType += loss_perType.cpu().detach().numpy()
            
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(val_dl):
                data = [d.to('cuda:0') for d in data]
                loss,loss_perType = model(*data,True)
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