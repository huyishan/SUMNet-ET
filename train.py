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
from scipy.misc import imsave

images_dir = '/home/data/huyishan/ThyroidData/data/'
labels_dir = '/home/data/huyishan/ThyroidData/groundtruth/'
trainDataLoader, validDataLoader = data_loaders(images_dir, labels_dir, bs=5)

vis = Visualizer('net-loss')

net = SUMNetbyET()
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
lr_schduler = optim.lr_scheduler.StepLR(optimizer,step_size=500,gamma = 0.8)
criterion = nn.BCELoss()



def train(trainDataLoader, validDataLoader, net, optimizer, scheduler, criterion, use_gpu):
    i = 0
    j = 0
    epochs = 50
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
            probs,preimg = net(inputs)
            loss = 0.8*criterion(probs.view(-1), labels.view(-1)) + 0.2*criterion(preimg.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            # ispreds_0 = torch.nonzero(preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # if(epoch==5):
            #     plot_n_save(inputs.cpu().detach().numpy(), labels.cpu().detach().numpy(), preimg.cpu().detach().numpy(), j)
            #     j = i*10
            #     i = i+1
            # if(epoch>5):
            #     return
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
            probs,_ = net(inputs)
            loss = criterion(probs.view(-1), labels.view(-1))
            preds = (probs > 0.5).float()
            validDice += dice_coefficient(preds, labels).item()
            validRunningLoss += loss.item()
            validBatches += 1
        validLoss.append(validRunningLoss / validBatches)
        validDiceCoeff.append(validDice / validBatches)
        if validDice >= bestValidDice:
            bestValidDice = validDice
            torch.save(net.state_dict(), 'SUMNetbyET.pt')
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

def plot_n_save(images, labels, preds, j):
    preds = (preds > 0.5) * 1
    num = images.shape[0]
    images = images * 255
    images = images.astype('uint8')
    labels = labels.astype('uint8')
    preds = preds.astype('uint8')
    for i in range(num):
        image = images[i][0]
        label = labels[i][0]
        pred = preds[i][0]
        thresh_label = cv2.threshold(label, 0.99, 255, cv2.THRESH_BINARY)[1]
        thresh_pred = cv2.threshold(pred, 0.99, 255, cv2.THRESH_BINARY)[1]
        contours_label = cv2.findContours(thresh_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours_pred = cv2.findContours(thresh_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        image_rgb = np.empty((256, 384, 3), dtype='uint8')
        image_rgb[:, :, 0] = image
        image_rgb[:, :, 1] = image
        image_rgb[:, :, 2] = image
        result = cv2.drawContours(image_rgb, contours_label, -1, (0, 255, 0), 1)
        result = result.astype('uint8')
        result = cv2.drawContours(result, contours_pred, -1, (255, 0, 0), 1)
        name1 = 'SUMNet/16/results1/a' + str(j) + '.png'
        name2 = 'SUMNet/16/results1/b' + str(j) + '.png'
        name3 = 'SUMNet/16/results1/c' + str(j) + '.png'
        kernel = np.ones((5, 5), np.uint8)
        label_erode = cv2.erode(label, kernel, iterations=1)
        result_label = (label - label_erode) * 255
        pred_erode = cv2.erode(pred, kernel, iterations=1)
        result_pred = (pred - pred_erode) * 255
        j = j + 1
        imsave(name1, result)
        imsave(name2, result_label)
        imsave(name3, result_pred)
    return

DF = train(trainDataLoader, validDataLoader, net, optimizer,lr_schduler, criterion, use_gpu)
DF.to_csv('SUMNetbyET.csv')
