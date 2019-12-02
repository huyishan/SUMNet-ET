#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:44:02 2018

@author: sumanthnandamuri
"""
import numpy as np
import torch
from SUMNet import SUMNet
import cv2
from get_data_loader import data_loaders
from ETNet import ETNet
from SUMNetbyET import SUMNetbyET

import os
from  dataset import USDataset
from torch.utils.data import DataLoader
import tqdm
import time

net = SUMNet()
net.load_state_dict(torch.load('SUMNet_3channel.pt'))
net.eval()
net = net.cuda()

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(file)
    return L

testimgnamelist = file_name("/home/data/huyishan/predata_thyroid/test/data")
root = "/home/data/huyishan/predata_thyroid/"
testdataset = USDataset(root=root,filename=testimgnamelist,mode='test')

testdl = DataLoader(testdataset,batch_size=10,shuffle=False,num_workers=2,pin_memory = True)

def plot_n_save(images, labels, preds, j):
    num = images.shape[0]
    images = images*255
    images = images.astype('uint8')
    labels = labels.astype('uint8')
    preds = preds.astype('uint8')
    for i in range(num):
        image = images[i][0]
        label = labels[i][0]
        pred = preds[i][0]
        thresh_label = cv2.threshold(label, 0.99, 255, cv2.THRESH_BINARY)[1]
        thresh_pred = cv2.threshold(pred, 0.99, 255, cv2.THRESH_BINARY)[1]
        contours_label = cv2.findContours(thresh_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_pred = cv2.findContours(thresh_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        image_rgb = np.empty((256, 384, 3), dtype ='uint8')
        image_rgb[:, :, 0] = images[i][0]
        image_rgb[:, :, 1] = images[i][1]
        image_rgb[:, :, 2] = images[i][2]
        result = cv2.drawContours(image_rgb, contours_label, -1, (0,255,0), 1)
        result = result.astype('uint8')
        result = cv2.drawContours(result, contours_pred, -1, (255,0,0), 1)
        name1 = './SUMNet/16/results/'+str(j)+'.png'
        name2 = './SUMNet/16/gt/'+str(j)+'.png'
        name3 = './SUMNet/16/preds/'+str(j)+'.png'
        kernel = np.ones((5, 5),np.uint8)
        label_erode = cv2.erode(label,kernel,iterations = 1)
        result_label = (label - label_erode)*255
        pred_erode = cv2.erode(pred,kernel,iterations = 1)
        result_pred = (pred - pred_erode)*255
        j = j+1
        cv2.imwrite(name1, result)
        cv2.imwrite(name2,result_label)
        cv2.imwrite(name3, result_pred)
    return


j = 0
i = 0
bar = tqdm.tqdm(testdl)
for data in bar:
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    logits = net(images)

    preds = logits.cpu().detach().numpy()
    preds = (preds > 0.5)*1
    plot_n_save(images.cpu().detach().numpy(), labels.cpu().detach().numpy(), preds, j)
    i = i+1
    j = 10*i
