import cv2 as cv
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from matplotlib import pyplot as plt
import pydicom

# 中间层类提取特征并保存
class FeatureVisualization():
    def __init__(self, img, selected_layer, model, shownum):
        self.img = img
        self.selected_layer = selected_layer
        self.model = model
        self.shownum = shownum
        self.input = img

    def get_features(self):
        x = self.input
        layer_model = self.model
        listlength = len(self.selected_layer);
        if listlength>1:
            for i in range(listlength -1):
                for name, layer in layer_model._modules.items():
                    if name == self.selected_layer[i]:
                        layer_model = layer_model._modules[self.selected_layer[i]]
                        break
                    x = layer(x)

        for name,layer in layer_model._modules.items():
            if name == self.selected_layer[-1]:
                return x
            x = layer(x)


    def save_features_to_imgs(self):
        show_feature = []
        features = self.get_features()
        showfeaturesnum = (self.shownum if (self.shownum < features.shape[1]) else features.shape[1])
        for i in range(showfeaturesnum):
            feature = features[:, i, :, :]
            feature = feature.view(feature.shape[1], feature.shape[2])
            feature = self.get_single_feature()
            feature = feature.data.cpu().detach().numpy()
            # feature = 1.0/(1+np.exp(-1*feature))
            # feature = np.round(feature*255)
            show_feature.append(feature)
        return show_feature

    def get_single_feature(self):
        features = self.get_features()
        feature = features[:, 0, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        return feature

    def save_feature_to_img(self):
        # to numpy
        feature = self.get_single_feature()
        feature = feature.data.numpy()
        # feature = 1.0 / (1 + np.exp(-1 * feature))
        # feature = np.round(feature * 255)
        print(feature[0])
        cv.imwrite('./featureimg.jpg', feature)

# 定义并读取数据
images_dir = '/home/data/huyishan/ThyroidData/data/D16.dcm'
labels_dir = '/home/data/huyishan/ThyroidData/groundtruth/D16.dcm'
images_vol = pydicom.read_file(images_dir).pixel_array
labels_vol = pydicom.read_file(labels_dir).pixel_array
begin = 0
end = 0
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

# 获取网络模型参数
from BASNet import BASNet
net = BASNet(1)
net.load_state_dict(torch.load('BASNet_CENet.pt'))
net.eval()
print(net)
net = net.cuda()

#选择要运行查看的输入图片
image = images_vol[1]
cv.imwrite("result/feature/orgin.jpg",image)
label = labels_vol[1]
cv.imwrite("result/feature/label.jpg",label*255)
image = image[np.newaxis,np.newaxis,:,:]
image = torch.from_numpy(image).float().cuda()

# 查看外层参数名的第几层层名数据及显示特征的数量
netnamelist = ['inconv']
myClass = FeatureVisualization(image,netnamelist,net,20)
fearure_imgs = myClass.save_features_to_imgs()
print(len(fearure_imgs))
x = len(fearure_imgs)/5
y = 5
plt.figure(figsize=(15,10)) #设置窗口大小
for i in range(1,len(fearure_imgs)+1):
    # cv.imwrite("result/feature/feature_"+netnamelist[-1]+"_"+str(i)+".jpg", fearure_imgs[i-1])
    plt.subplot(x,y,i)
    plt.imshow(fearure_imgs[i-1],cmap='gray')
    plt.axis('off')
plt.show()
print("")
