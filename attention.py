#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:00:54 2019

@author: will
"""

from torch.utils.data import Dataset
import torch

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
    # batch is a list of things returned by __getitem__.
    # e.g. in this case a list of tuple of three element
    # this function should return a batch of data
    if len(batch[0]) == 3:
        node,edge,y = zip(*batch)
    else:
        node,edge = zip(*batch)
    out = torch.nn.utils.rnn.pad_sequence(node)
    mask = (out==0).all(2).T
    edge = torch.stack(edge)[None,:,:]
    y = torch.stack(y)
    return out,mask,edge,y    
