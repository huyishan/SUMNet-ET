#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:18:58 2018

@author: sumanthnandamuri
"""
import os
import pydicom
import imageio
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skimage.transform import resize
import cv2
import warnings
warnings.filterwarnings('ignore')

def extract_data(images_dir, images_files_list, labels_dir, labels_files_list, is_train = True):    
    images_batches = []
    labels_batches = []    
    for images_file, labels_file in zip(images_files_list, labels_files_list):
        begin = 0
        end = 0
        images_vol = pydicom.read_file(images_dir+images_file).pixel_array
        labels_vol = pydicom.read_file(labels_dir+labels_file).pixel_array
        if is_train:
            n = len(images_vol)//2 - 1        
            for i in range(0, n, 1):
                if labels_vol[i].sum() <= 1500:
                    begin += 1
                else:
                    break                
            for i in range(n, 0, -1):
                if labels_vol[i].sum() <= 1500:
                    end += 1
                else:
                    break                
            images_vol = images_vol[begin:n-end+1]
            labels_vol = labels_vol[begin:n-end+1]
        images_vol = resize(images_vol, (len(images_vol), 256, 384))
        labels_vol = resize(labels_vol, (len(labels_vol), 256, 384))
        images_batches.append(images_vol)
        labels_batches.append(labels_vol)
    return images_batches, labels_batches

# 将灰度图转换为一个三通道的图
def grayto3channel(grayimgs):
    contactimgs = []
    for i in range(len(grayimgs)):
        grayimg = grayimgs[i]
        grayimg = cv2.resize(grayimg,(256,384))
        #将灰度图进行限制性直方图均衡处理,限制对比度设置为2
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        limit_gray = clahe.apply(grayimg)
        #高通滤波器选用双边滤波器，获取高频信息
        gray_H = cv2.bilateralFilter(limit_gray,0,100,15)
        #低通滤波采用高斯模糊，获取低频信息
        gray_L = cv2.GaussianBlur(limit_gray,(15,15),0)
        #增加通道维度
        gray_channel = grayimg[np.newaxis, :, :]
        limit_gray_channel = limit_gray[np.newaxis, :, :]
        gray_H_channel = gray_H[np.newaxis, :, :]
        gray_L_channel = gray_L[np.newaxis, :, :]
        #数组整合
        gray_merge = np.concatenate((gray_channel,gray_H_channel,gray_L_channel),axis=0)
        contactimgs.append(gray_merge)
    contactimgs = np.array(contactimgs)
    return contactimgs

def labelsersize(labelimgs):
    contactimgs = []
    for i in range(len(labelimgs)):
        grayimg = labelimgs[i]
        grayimg = cv2.resize(grayimg, (256, 384))
        contactimgs.append(grayimg)
    contactimgs = np.array(contactimgs)
    return contactimgs
def data_loaders(images_dir, labels_dir, bs = 12):
    images_files_list = os.listdir(images_dir)
    labels_files_list = os.listdir(labels_dir)
    images_files_list.sort()
    labels_files_list.sort()
    trainimages_files_list = images_files_list[:15]
    trainlabels_files_list = labels_files_list[:15]
    validimages_files_list = images_files_list[15:16]
    validlabels_files_list = labels_files_list[15:16]
    trainimages_batches, trainlabels_batches = extract_data(images_dir, trainimages_files_list, labels_dir, trainlabels_files_list, is_train = True)
    validimages_batches, validlabels_batches = extract_data(images_dir, validimages_files_list, labels_dir, validlabels_files_list, is_train = True)
    trainimages_batches = np.array(trainimages_batches)
    trainlabels_batches = np.array(trainlabels_batches)    
    validimages_batches = np.array(validimages_batches)
    validlabels_batches = np.array(validlabels_batches)
    
    trainimages = trainimages_batches[0].reshape((len(trainimages_batches[0]), 1, 256, 384))
    trainlabels = trainlabels_batches[0].reshape((len(trainlabels_batches[0]), 1, 256, 384))
    for i in range(1, len(trainimages_batches)):
        n = len(trainimages_batches[i])
        trainimages = np.concatenate((trainimages, trainimages_batches[i].reshape((n, 1, 256, 384))), axis = 0)
        trainlabels = np.concatenate((trainlabels, trainlabels_batches[i].reshape((n, 1, 256, 384))), axis = 0)
    validimages = validimages_batches[0].reshape((len(validimages_batches[0]), 1, 256, 384))
    validlabels = validlabels_batches[0].reshape((len(validlabels_batches[0]), 1, 256, 384))
    for i in range(1, len(validimages_batches)):
        n = len(validimages_batches[i])
        validimages = np.concatenate((validimages, validimages_batches[i].reshape((n, 1, 256, 384))), axis = 0)
        validlabels = np.concatenate((validlabels, validlabels_batches[i].reshape((n, 1, 256, 384))), axis = 0)
    trainlabels = (trainlabels*255 >0)*1
    validlabels = (validlabels*255 >0)*1    
    flippedtrainimages = np.flip(trainimages, axis = 3)
    flippedtrainlabels = np.flip(trainlabels, axis = 3)
    trainimages = np.concatenate((trainimages, flippedtrainimages), axis = 0)
    trainlabels = np.concatenate((trainlabels, flippedtrainlabels), axis = 0)
    
    batch_size   = bs
    trainimages  = torch.from_numpy(trainimages).float()
    trainlabels  = torch.from_numpy(trainlabels).float()
    trainDataset = TensorDataset(trainimages, trainlabels)
    trainDataLoader = DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)    
    validimages = torch.from_numpy(validimages).float()
    validlabels = torch.from_numpy(validlabels).float()
    validDataset = TensorDataset(validimages, validlabels)
    validDataLoader = DataLoader(validDataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    return trainDataLoader, validDataLoader    
