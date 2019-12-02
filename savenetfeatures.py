import cv2 as cv
import numpy as np
import torch
from torchvision import models
from torch.autograd import Variable

class FeatureVisualization():
    def __init__(self,img,selected_layer,model):
        self.img = img
        self.selected_layer = selected_layer
        self.model = model

    def get_feature(self):
        input = self.img
        print(input.shape)
        x = input
        for index,layer in enumerate(self.model):
            x = layer(x)
            if(index == self.selected_layer):
                return x

    def get_single_feature(self):
        features = self.get_feature()
        print(features.shape)

        feature = features[:,0,:,:]
        print(feature.shape)

        feature = feature.view(feature.shape[1],feature[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        # to numpy
        feature = self.get_single_feature()
        feature = feature.data.numpy()

        feature = 1.0/(1+np.exp(-1*feature))
        feature = np.round(feature*255)
        print(feature[0])
        cv.imwrite('./featureimg.jpg',feature)

if __name__=='__main__':
    myClass = FeatureVisualization("./a.jpg",5)
    print(myClass.model)

    myClass.save_feature_to_img()