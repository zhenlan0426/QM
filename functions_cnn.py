#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:13:24 2019

@author: will
"""
#import torchvision.transforms as transforms
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import copy
import time



def process_data(train,test,type_):
    train_ids = train.loc[train.type==type_]
    train_ids['file_name'] = train_ids.molecule_name.str.cat(train_ids.id.astype(str),sep='_')
    test_ids = test.loc[test.type==type_]
    test_ids['file_name'] = test_ids.molecule_name.str.cat(test_ids.id.astype(str),sep='_')
    return train_ids,test_ids

class CustomImageDataset(Dataset):
   
    def __init__(self, id_df, root_dir, transform=None, IsTrain=True):
        # id_df is a list of images to load
        # target_dict is a dict, with id as key and target as value
        self.id_df = id_df
        self.root_dir = root_dir
        self.transform = transform
        self.IsTrain = IsTrain
        
    def __len__(self):
        return self.id_df.shape[0]

    def __getitem__(self, idx):
        
        img_name = os.path.join(
            self.root_dir,
            self.id_df.iloc[idx]['file_name'] +'.pkl'
            #str(self.df.iloc[idx]['molecule_name']) + '_' + str(self.df.iloc[idx]['id']) + '.pkl'
        )
        
        with open (img_name, 'rb') as fp:
            image = pickle.load(fp)
        
        for c in range(5):
            image[c] = np.clip(image[c], 0, 255) / 255
        
        img = torch.from_numpy(np.array(image))
        img = img.type(torch.FloatTensor)
        
        if self.transform:
            img = self.transform(img)

        if self.IsTrain:
            return img, self.id_df.iloc[idx]['scalar_coupling_constant']
        else:
            return img
        
class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(30976, 1),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class CNN2(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN2, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear = nn.Sequential(nn.Linear(256, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,1),
                                    nn.ReLU(inplace=True),
                                    )
        
        self.fc_layer = nn.Sequential(nn.Linear(121, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,1),
                                    )


    def forward(self, x):
        n = x.shape[0]
        x = self.conv_layer(x).transpose(1,3)
        x = self.linear(x).reshape(n,-1)
        x = self.fc_layer(x)
        return x

def train_cnn(model,optimizer,train_loader,valid_loader,n_epochs,clip,scheduler):
    start_time = time.time()
    criterion = nn.L1Loss()

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