
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
sys.path.insert(0,'/data0/qilei_chen/Development/vision2')
import torchvision
from torchvision import datasets, models, transforms
#from networks import *
from torch.autograd import Variable
from PIL import Image
import glob
import cv2
import datetime
import time
def micros(t1, t2):
    delta = (t2-t1).microseconds
    return delta



class classifier:
    def __init__(self,input_size_=1000,mean_=[0.485, 0.456, 0.406],std_=[0.229, 0.224, 0.225],class_num_=2,model_name = 'resnet101_wide',device_id=0):
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
        self.device = torch.device("cuda:"+str(device_id))
        if model_name == "alexnet":
            """ Alexnet
            """
            self.model = models.alexnet()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            self.model = models.vgg11_bn()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "vgg11":
            """ VGG11_bn
            """
            self.model = models.vgg11()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,self.class_num)
            input_size = 224
        elif model_name == "squeezenet1_0":
            """ squeezenet1_0
            """
            self.model = models.squeezenet1_0()
            #set_parameter_requires_grad(model_ft, feature_extract)
            self.model.classifier[1] = nn.Conv2d(512, self.class_num, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.class_num
            input_size = 224
        elif model_name == "resnet":
            """ Resnet18
            """
            self.model = models.resnet18()
            #set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.class_num)
            input_size = 224

        elif model_name=='inception_v3':
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
        elif model_name == "densenet121":
            """ Densenet
            """
            self.model = models.densenet121()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num) 
        elif model_name == "densenet161":
            """ Densenet
            """
            self.model = models.densenet161()
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.class_num)             
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    def ini_model(self,model_dir):
        checkpoint = torch.load(model_dir)
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint)
        
        #self.model.cuda()
        self.model.to(self.device)
        #print(self.model)
        cudnn.benchmark = True
        self.model.eval()
    def predict(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image = Image.open(img_dir).convert('RGB')
        image = Image.fromarray(img)
        t1 = datetime.datetime.now()
        image = self.test_transform(image)
        inputs = image
        inputs = Variable(inputs, volatile=True)
        
        inputs = inputs.to(self.device)
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front

        outputs = self.model(inputs)
        softmax_res = self.softmax(outputs.data.cpu().numpy()[0])
        probilities = []
        for probility in softmax_res:
            probilities.append(probility)
        t2 = datetime.datetime.now()
        #print(micros(t1,t2)/1000)
        return probilities.index(max(probilities))
    def predict1(self,img_dir):
        img = cv2.imread(img_dir)
        return self.predict(img)

def process_4_situation_videos(model_name = "densenet161"):
    

    model = classifier(224,model_name=model_name,class_num_=4)
    
    model_dir = '/data2/qilei_chen/DATA/GI_4_NEW/finetune_4_new_oimg_'+model_name+'/best.model'

    model.ini_model(model_dir)

    videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/preprocessed/"
    #videos_folder = "/data2/qilei_chen/jianjiwanzhengshipin2/weijingshi4/"
    '''
    big_roi = [441, 1, 1278, 720]
    small_roi = [156, 40, 698, 527]

    roi = big_roi
    '''
    video_start = -1#15

    videos_result_folder = os.path.join(videos_folder,"result_"+model_name)

    video_suffix = ".avi"
    
    video_file_dir_list = glob.glob(os.path.join(videos_folder,"*"+video_suffix))

    if not os.path.exists(videos_result_folder):
        os.makedirs(videos_result_folder)
    video_count=0
    for video_file_dir in video_file_dir_list:

        if video_count>video_start:
            count=1

            video = cv2.VideoCapture(video_file_dir)

            success,frame = video.read()
        
            video_name = os.path.basename(video_file_dir)

            records_file_dir = os.path.join(videos_result_folder,video_name.replace(video_suffix,".txt"))
            records_file_header = open(records_file_dir,"w")

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            show_result_video_dir = os.path.join(videos_result_folder,video_name)
            #videoWriter = cv2.VideoWriter(show_result_video_dir,cv2.VideoWriter_fourcc("P", "I", "M", "1"),fps,frame_size)
            print(show_result_video_dir)
            while success:
                '''
                frame_roi = frame[roi[1]:roi[3],roi[0]:roi[2]]
                predict_label = model.predict(frame_roi)
                '''
                predict_label = model.predict(frame)
                records_file_header.write(str(count)+" "+str(predict_label)+"\n")
                #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame_roi)
                cv2.putText(frame,str(count)+":"+str(predict_label),(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
                #cv2.imwrite("/data2/qilei_chen/DATA/test.jpg",frame)
                #videoWriter.write(frame)
                #print(predict_label)
                success,frame = video.read()
                count+=1
            
        video_count+=1
'''
process_4_situation_videos(model_name='alexnet')
process_4_situation_videos(model_name='vgg11')
process_4_situation_videos(model_name='vgg13')
process_4_situation_videos(model_name='vgg16')
process_4_situation_videos(model_name='vgg19')
'''

process_4_situation_videos(model_name='squeezenet1_0')
process_4_situation_videos(model_name='vgg11_bn')
process_4_situation_videos(model_name='vgg13_bn')
process_4_situation_videos(model_name='vgg16_bn')
process_4_situation_videos(model_name='vgg19_bn')
'''
model_name='densenet121'
cf = classifier(224,model_name=model_name,class_num_=4)
#lesion_category = 'Cotton_Wool_Spot'
folder_label = 3
#model_dir = '/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/models_4_'+lesion_category+'/densenet_epoch_16.pth'
model_dir = '/data2/qilei_chen/DATA/4class_c/finetune_4_end0_'+model_name+'/best.model'
cf.ini_model(model_dir)
#for i in range(100):
#image_file_dirs = glob.glob('/data0/qilei_chen/Development/Datasets/DR_LESION_PATCH/'+lesion_category+'/val/'+str(folder_label)+'/*.jpg')
image_file_dirs = glob.glob('/data2/qilei_chen/DATA/4class_c/val/'+str(folder_label)+'/*.jpg')
#print(image_file_dirs)
#count = 0
wrong_count=0
count = [0,0,0,0,0]
print('groundtruth:'+str(folder_label))
for image_file_dir in image_file_dirs:
    #print(image_file_dir)
    label = cf.predict1(image_file_dir)
    
    if label!=folder_label:
        print(label)
        print(image_file_dir)
        pass
    #'
    #    wrong_count+=1
    #    #cv2.imshow('test',cv2.imread(image_file_dir))
    #    #cv2.waitKey(0)
    #count += 1
    #'
    count[int(label)]+=1
print(count)
'''

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

