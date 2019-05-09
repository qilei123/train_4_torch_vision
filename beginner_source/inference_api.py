
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import time
import copy
import os
import sys
import argparse
import csv
sys.path.insert(0,'/media/cql/DATA0/Development/torch_vision')
import torchvision
from torchvision import datasets, models, transforms
#from networks import *
from torch.autograd import Variable
from PIL import Image

class classifier:
    def __init__(self,input_size_=1000,mean_=[0.485, 0.456, 0.406],std_=[0.229, 0.224, 0.225],class_num_=2):
        self.input_size = input_size_
        self.mean = mean_
        self.std = std_
        self.test_transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        self.class_num = class_num_
        self.model = models.inception_v3(pretrained=True)
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.class_num)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,self.class_num)

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def ini_model(self,model_dir):
        checkpoint = torch.load(model_dir,map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cuda()
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

cf = classifier()
cf.ini_model('/home/cql/Downloads/binary_models/inception_epoch_6_1000.pth')
for i in range(100):    
    print(cf.predict('/home/cql/Downloads/test5.7/test/13_left.jpeg'))