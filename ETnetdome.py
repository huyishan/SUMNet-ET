import torch
from torchvision import models,transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# testtrnsor = torch.rand(1,3,512,512)
# print(testtrnsor.shape)
# preconv = nn.Conv2d(3,64,1)
# testtrnsor = preconv(testtrnsor)
# resnet50 = models.resnet50(pretrained=True)
# encode1 = resnet50.layer1[0]
# encode2 = resnet50.layer2[0]
# encode3 = resnet50.layer3[0]
# encode4 = resnet50.layer4[0]
# testtrnsor_e1 = encode1(testtrnsor)
# testtrnsor_e2 = encode2(testtrnsor_e1)
# testtrnsor_e3 = encode3(testtrnsor_e2)
# testtrnsor_e4 = encode4(testtrnsor_e3)
#
# decode1_c1 =  nn.Conv2d(2048,1024,1)
# decode1_up = nn.ConvTranspose2d(1024,1024,2,2)
# decode1_c2 = nn.Sequential(nn.Conv2d(1024,512,1),nn.Conv2d(512,512,3,1,1),nn.Conv2d(512,1024,1))
# testtrnsor_d1_c1 = decode1_c1(testtrnsor_e4)
# testtrnsor_d1_up = decode1_up(testtrnsor_d1_c1)
# testtrnsor_d1_c2 = decode1_c2(torch.add(testtrnsor_d1_up,testtrnsor_e3))
#
# decode2_c1 =  nn.Conv2d(1024,512,1)
# decode2_up = nn.ConvTranspose2d(512,512,2,2)
# decode2_c2 = nn.Sequential(nn.Conv2d(512,256,1),nn.Conv2d(256,256,3,1,1),nn.Conv2d(256,512,1))
#
# testtrnsor_d2_c1 = decode2_c1(testtrnsor_d1_c2)
# testtrnsor_d2_up = decode2_up(testtrnsor_d2_c1)
# testtrnsor_d2_c2 = decode2_c2(torch.add(testtrnsor_d2_up,testtrnsor_e2))
#
#
# decode3_c1 =  nn.Conv2d(512,256,1)
# decode3_up = nn.ConvTranspose2d(256,256,2,2)
# decode3_c2 = nn.Sequential(nn.Conv2d(256,128,1),nn.Conv2d(128,128,3,1,1),nn.Conv2d(128,256,1))
#
# testtrnsor_d3_c1 = decode3_c1(testtrnsor_d2_c2)
# testtrnsor_d3_up = decode3_up(testtrnsor_d3_c1)
# testtrnsor_d3_c2 = decode3_c2(torch.add(testtrnsor_d3_up,testtrnsor_e1))
# print(testtrnsor_d1_c2.shape)
# print(testtrnsor_d2_c2.shape)
# print(testtrnsor_d3_c2.shape)

# d1 = torch.rand(1,1024,128,128)
# d2 = torch.rand(1,512,256,256)
# d3 = torch.rand(1,256,512,512)
#
# wb1_c1 = nn.Conv2d(1024,1024,1)
# d1_c = wb1_c1(d1)
# wb1_ga = nn.AdaptiveAvgPool2d(1)
# d1_ga = wb1_ga(d1_c)
# wb1_c2 = nn.Sequential(nn.Conv2d(1024,1024,1),nn.ReLU(),nn.Conv2d(1024,1024,1),nn.Sigmoid())
# d1_c2 = wb1_c2(d1_ga)
# d1 = torch.mul(d1_c2,d1_c)
# wb1_up = nn.ConvTranspose2d(1024,512,2,2)
# d1_up = wb1_up(d1)
# print(d1_up.shape)
#
# wb2_c1 = nn.Conv2d(512,512,1)
# d2_c = wb2_c1(d2)
# wb2_ga = nn.AdaptiveAvgPool2d(1)
# d2_ga = wb1_ga(d2_c)
# wb2_c2 = nn.Sequential(nn.Conv2d(512,512,1),nn.ReLU(),nn.Conv2d(512,512,1),nn.Sigmoid())
# d2_c2 = wb2_c2(d2_ga)
# d2 = torch.mul(d2_c2,d2_c)
# wb2_up = nn.ConvTranspose2d(512,256,2,2)
# d2_up = wb2_up(torch.add(d2,d1_up))
# # torch.Size([1, 256, 512, 512])
#
#
# wb3_c1 = nn.Conv2d(256,256,1)
# d3_c = wb3_c1(d3)
# wb3_ga = nn.AdaptiveAvgPool2d(1)
# d3_ga = wb3_ga(d3_c)
# wb3_c2 = nn.Sequential(nn.Conv2d(256,256,1),nn.ReLU(),nn.Conv2d(256,256,1),nn.Sigmoid())
# d3_c2 = wb3_c2(d3_ga)
# d3 = torch.mul(d3_c2,d3_c)
# d3_add = torch.add(d3,d2_up)
#
# print(d3_add.shape)




e1 = torch.rand(1,256,512,512)
e2 = torch.rand(1,512,256,256)

egm1 = nn.Sequential(nn.Conv2d(256,256,1),nn.Conv2d(256,256,3,1,1))
egm2 = nn.Sequential(nn.ConvTranspose2d(512,256,2,2),nn.Conv2d(256,256,1),nn.Conv2d(256,256,3,1,1))
egm_cat = torch.cat([egm1(e1),egm2(e2)],1)
print(egm_cat.shape)
print(nn.Conv2d(512,256,1)(egm_cat))
# nn.Conv2d(512,1,1)(egm_cat)
