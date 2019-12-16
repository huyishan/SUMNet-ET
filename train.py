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
from BASNet import BASNet

import pytorch_iou
import pytorch_ssim

images_dir = '/home/data/huyishan/ThyroidData/data/'
labels_dir = '/home/data/huyishan/ThyroidData/groundtruth/'
trainDataLoader, validDataLoader = data_loaders(images_dir, labels_dir, bs=5)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	return loss0, loss


vis = Visualizer('net-loss')

net = BASNet(1)
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
lr_schduler = optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma = 0.8)
criterion = nn.BCELoss()
bce_loss = nn.BCELoss(size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def train(trainDataLoader, validDataLoader, net, optimizer, scheduler, criterion, use_gpu):
    i = 0
    j = 0
    epochs = 10
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
            # islabels_0 = torch.nonzero(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            d0 = net(inputs)
            loss = bce_ssim_loss(d0,labels)
            # loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels)
            preds = (d0 > 0.5).float()

            # probs = net(inputs)
            # loss = criterion(probs.view(-1), labels.view(-1))

            # preds = (probs > 0.5).float()
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
                p['lr'] *= 0.8
        trainLoss.append(trainRunningLoss / trainBatches)
        trainDiceCoeff.append(trainDice / trainBatches)

        net.train(False)
        for data in validDataLoader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            d0= net(inputs)
            loss = bce_ssim_loss(d0,labels)
            # loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels)
            preds = (d0 > 0.5).float()

            # probs = net(inputs)
            # loss = criterion(probs.view(-1), labels.view(-1))
            # preds = (probs > 0.5).float()
            validDice += dice_coefficient(preds, labels).item()
            validRunningLoss += loss.item()
            validBatches += 1
        validLoss.append(validRunningLoss / validBatches)
        validDiceCoeff.append(validDice / validBatches)
        if validDice >= bestValidDice:
            bestValidDice = validDice
            torch.save(net.state_dict(), 'BASNet.pt')
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


DF = train(trainDataLoader, validDataLoader, net, optimizer,lr_schduler, criterion, use_gpu)
DF.to_csv('BASNet.csv')
