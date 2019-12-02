#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:27:04 2018

@author: sumanthnandamuri
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import tqdm
import time
import gc
from get_data_loader import data_loaders
from SUMNetbyET import SUMNetbyET
from SUMNet import SUMNet
from utils import dice_coefficient, make_loader
from ETNet import ETNet
from visualize import Visualizer
import cv2

from dataset import USDataset
import os
from torch.utils.data import DataLoader


def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(file)
    return L

trainimgnamelist = file_name("/home/data/huyishan/predata_thyroid/train/data")
testimgnamelist = file_name("/home/data/huyishan/predata_thyroid/test/data")
root = "/home/data/huyishan/predata_thyroid/"

traindataset = USDataset(root=root,filename=trainimgnamelist,mode='train')
testdataset = USDataset(root=root,filename=testimgnamelist,mode='test')

traindl = DataLoader(traindataset,batch_size=10,shuffle=True,num_workers=2,pin_memory = True)
testdl = DataLoader(testdataset,batch_size=10,shuffle=False,num_workers=2,pin_memory = True)

# dl_data = iter(testdl)
# print(next(dl_data))

vis = Visualizer('net-loss')
net = SUMNet()
# use_gpu = torch.cuda.is_available()
use_gpu = True
if use_gpu:
    net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
lr_schduler = optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma = 0.8)
criterion = nn.BCELoss()



def train(trainDataLoader, validDataLoader, net, optimizer, scheduler, criterion, use_gpu):
    i = 0
    j = 0
    epochs = 20
    trainLoss = []
    validLoss = []
    trainDiceCoeff = []
    validDiceCoeff = []
    start = time.time()
    bestValidDice = 0
    for epoch in range(epochs):
        epochStart = time.time()
        trainRunningLoss = 0
        validRunningLoss = 0
        trainBatches = 0
        validBatches = 0
        trainDice = 0
        validDice = 0

        net.train(True)
        bar = tqdm.tqdm(trainDataLoader)
        for data in bar:
            inputs, labels = data
#             # islabels_0 = torch.nonzero(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            probs = net(inputs)
            loss = criterion(probs.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            # ispreds_0 = torch.nonzero(preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainRunningLoss += loss.item()
            trainDice += dice_coefficient(preds, labels).item()
            trainBatches += 1
            vis.plot("lr",optimizer.state_dict()['param_groups'][0]['lr'])
            vis.plot("loss", loss.item())
            bar.set_postfix(loss=loss.item())
        if epoch>0 and epoch%5==0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        trainLoss.append(trainRunningLoss / trainBatches)
        trainDiceCoeff.append(trainDice / trainBatches)

        net.train(False)
        for data in validDataLoader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            probs = net(inputs)
            loss = criterion(probs.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            validDice += dice_coefficient(preds, labels).item()
            validRunningLoss += loss.item()
            validBatches += 1
        validLoss.append(validRunningLoss / validBatches)
        validDiceCoeff.append(validDice / validBatches)
        if validDice >= bestValidDice:
            bestValidDice = validDice
            torch.save(net.state_dict(), 'SUMNet_3channel.pt')
        epochEnd = time.time() - epochStart
        print('Epoch: {:.0f}/{:.0f} | Train Loss: {:.3f} | Valid Loss: {:.3f} | Train Dice: {:.3f} | Valid Dice: {:.3f}' \
              .format(epoch + 1, epochs, trainRunningLoss / trainBatches, validRunningLoss / validBatches,
                      trainDice / trainBatches, validDice / validBatches))
        print('Time: {:.0f}m {:.0f}s'.format(epochEnd // 60, epochEnd % 60))
    end = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(end // 60, end % 60))
    trainLoss = np.array(trainLoss)
    validLoss = np.array(validLoss)
    trainDiceCoeff = np.array(trainDiceCoeff)
    validDiceCoeff = np.array(validDiceCoeff)
    DF = pd.DataFrame(
        {'Train Loss': trainLoss, 'Valid Loss': validLoss, 'Train Dice': trainDiceCoeff, 'Valid Dice': validDiceCoeff})
    return DF



DF = train(traindl, testdl, net, optimizer,lr_schduler, criterion, use_gpu)
DF.to_csv('SUMNetbyET.csv')
