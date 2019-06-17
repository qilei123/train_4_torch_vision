
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import time
import copy
import os
import sys
import argparse
import csv
sys.path.insert(0,'/media/cql/DATA1/Development/vision2')
sys.path.insert(0,'/data0/qilei_chen/vision')
import torchvision
from torchvision import datasets, models, transforms
#from networks import *
from torch.autograd import Variable
from PIL import Image
import glob
import cv2

class classifier:
    def __init__(self,input_size_=1000,mean_=[0.485, 0.456, 0.406],std_=[0.229, 0.224, 0.225],class_num_=2,model_name = 'resnet101_wide'):
        self.input_size = input_size_
        self.mean = mean_
        self.std = std_
        self.test_transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                #transforms.Normalize(self.mean, self.std)
            ])
        self.class_num = class_num_
        if model_name=='inception_v3':
            self.model = models.inception_v3()
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name=='inception_v3_wide':
            self.model = models.inception_v3_wide()
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name=='resnet101_wide':
            self.model = models.resnet101_wide()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.class_num)
        elif model_name == "densenet":
            """ Densenet
            """
            self.model = models.densenet121()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
            
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def ini_model(self,model_dir):
        checkpoint = torch.load(model_dir,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()
        print(self.model)
        cudnn.benchmark = True
        self.model.eval()
    def predict(self,img_dir):
        image = Image.open(img_dir).convert('RGB')
        image = self.test_transform(image)
        inputs = image
        inputs = Variable(inputs, volatile=True)
        
        inputs = inputs.cuda()
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front

        outputs = self.model(inputs)
        softmax_res = self.softmax(outputs.data.cpu().numpy()[0])
        probilities = []
        for probility in softmax_res:
            probilities.append(probility)
        return probilities.index(max(probilities))

cf = classifier(1000,model_name='inception_v3',class_num_=5)
#lesion_category = 'Cotton_Wool_Spot'
folder_label = 3
#model_dir = '/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/models_4_'+lesion_category+'/densenet_epoch_16.pth'
model_dir = '/data0/qilei_chen/Development/Datasets/KAGGLE_DR/models_1000/inception_epoch_13.pth'
cf.ini_model(model_dir)
#for i in range(100):
#image_file_dirs = glob.glob('/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/val/'+str(folder_label)+'/*.jpg')
image_file_dirs = glob.glob('/data0/qilei_chen/Development/Datasets/KAGGLE_DR/val/'+str(folder_label)+'/*.jpeg')
#print(image_file_dirs)
#count = 0
wrong_count=0
count = [0,0,0,0,0]
print('groundtruth:'+str(folder_label))
for image_file_dir in image_file_dirs:
    label = cf.predict(image_file_dir)
    '''
    if label!=folder_label:
        print(label)
        wrong_count+=1
        #cv2.imshow('test',cv2.imread(image_file_dir))
        #cv2.waitKey(0)
    count += 1
    '''
    count[int(label)]+=1
print(count)
'''
print(cf.predict('/home/cql/Downloads/test5.7/test/16_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/172_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/217_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/286_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test/508_left.jpeg'))

print(cf.predict('/home/cql/Downloads/test5.7/test0/13_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/22_left.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/31_right.jpeg'))
print(cf.predict('/home/cql/Downloads/test5.7/test0/40_right.jpeg'))
'''