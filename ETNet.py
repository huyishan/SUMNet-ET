import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ETNet(nn.Module):
    def __init__(self):
        super(ETNet,self).__init__()
        self.preconv = nn.Conv2d(1,64,1)
        self.encoder = models.resnet50(pretrained=True)
        self.encode1 = self.encoder.layer1[0]
        self.encode2 = self.encoder.layer2[0]
        self.encode3 = self.encoder.layer3[0]
        self.encode4 = self.encoder.layer4[0]
        self.decode1 = D_Block(2048,512)
        self.decode2 = D_Block(1024,256)
        self.decode3 = D_Block(512,128)
        self.egm = EGM()
        self.wam = WAM()
        self.outputconv = nn.Conv2d(512,1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        preconv = F.relu(self.preconv(x), inplace=True)
        eblock1 = self.encode1(preconv)
        eblock2 = self.encode2(eblock1)
        eblock3 = self.encode3(eblock2)
        eblock4 = self.encode4(eblock3)
        decode1 = self.decode1(eblock4,eblock3)
        decode2 = self.decode2(decode1,eblock2)
        decode3 = self.decode3(decode2,eblock1)
        egm_output, egm_guidance = self.egm(eblock1,eblock2)
        weight_add = self.wam(decode1,decode2,decode3)
        output = self.outputconv(torch.cat([weight_add,egm_guidance],1))
        output = 0.9*output + 0.1 * egm_output
        output = torch.sigmoid(output)
        return output


class D_Block(nn.Module):
    def __init__(self,in_ch,block_ch):
        super(D_Block,self).__init__()
        self.d_block_c = nn.Conv2d(in_ch,in_ch//2,1)
        self.d_block_up = nn.ConvTranspose2d(in_ch//2,in_ch//2,2,2)
        self.d_block_s = nn.Sequential(nn.Conv2d(in_ch//2,block_ch,1),nn.Conv2d(block_ch,block_ch,3,1,1),nn.Conv2d(block_ch,in_ch//2,1))

    def forward(self, high_input,low_input):
        d_block_c = self.d_block_c(high_input)
        d_block_up = self.d_block_up(d_block_c)
        D_block = self.d_block_s(torch.add(d_block_up,low_input))
        return D_block

class EGM(nn.Module):
    def __init__(self):
        super(EGM,self).__init__()
        self.egm1 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.Conv2d(256, 256, 3, 1, 1))
        self.egm2 = nn.Sequential(nn.ConvTranspose2d(512,256,2,2),nn.Conv2d(256,256,1),nn.Conv2d(256,256,3,1,1))
        self.output = nn.Conv2d(512,1,1)
        self.guidance = nn.Conv2d(512,256,1)

    def forward(self, encode1,encode2):
        egm1 = self.egm1(encode1)
        egm2 = self.egm2(encode2)
        egm_cat = torch.cat([egm1,egm2],1)
        egm_output = self.output(egm_cat)
        egm_guidance = self.guidance(egm_cat)
        return egm_output,egm_guidance


class WAM(nn.Module):
    def __init__(self):
        super(WAM,self).__init__()
        self.wb1 = Weight_Block(1024)
        self.wb2 = Weight_Block(512)
        self.wb3 = Weight_Block(256)
        self.wb1_up = nn.ConvTranspose2d(1024,512,2,2)
        self.wb2_up = nn.ConvTranspose2d(512,256,2,2)


    def forward(self, decode1,decode2,decode3):
        wb1 = self.wb1(decode1)
        wb2 = self.wb2(decode2)
        wb3 = self.wb3(decode3)
        wb1_up = self.wb1_up(wb1)
        wb2_up = self.wb2_up(torch.add(wb1_up,wb2))
        wb_add = torch.add(wb3,wb2_up)
        return wb_add


class Weight_Block(nn.Module):
    def __init__(self,in_ch):
        super(Weight_Block,self).__init__()
        self.wb_c1 = nn.Conv2d(in_ch,in_ch,1)
        self.wb_c2 = nn.Conv2d(in_ch,in_ch,1)
        self.wb_c3 = nn.Conv2d(in_ch,in_ch,1)
        self.wb_gp = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        wb_c1 = self.wb_c1(input)
        wb_gp = self.wb_gp(wb_c1)
        wb_relu = F.relu(self.wb_c2(wb_gp),inplace=True)
        wb_sigmoid = F.sigmoid(self.wb_c3(wb_relu))
        output = torch.mul(wb_sigmoid,wb_c1)
        return output
