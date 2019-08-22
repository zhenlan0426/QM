#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:10:25 2019

@author: will
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_


def train_cnn(model,optimizer,train_loader,valid_loader,n_epochs,clip,scheduler):
    start_time = time.time()
    criterion = nn.SmoothL1Loss()

    valid_loss_min = np.Inf # track change in validation loss
    for epoch in range(1, n_epochs+1):
        
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for ind, (data, target) in enumerate(train_loader):
            print(ind, end='\r')
            
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output.view(data.shape[0]), target.float())
            loss.backward()
            clip_grad_value_(model.parameters(),clip)
            optimizer.step()
        
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)    
                loss = criterion(output.view(data.shape[0]), target.float())    
                valid_loss += loss.item() * data.size(0)
     
        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
    
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            bestWeight = copy.deepcopy(model.state_dict())
        
        scheduler.step(valid_loss)
    
    model.load_state_dict(bestWeight)
    time_elapsed = time.time() - start_time
    print('Training completed in {}s'.format(time_elapsed))        
    
    return model