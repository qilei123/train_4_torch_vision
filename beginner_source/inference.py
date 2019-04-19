# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/fine-tuning.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : inference.py
# The main code for inference test phase of trained model.
# ***********************************************************

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
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

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--net_type', default='inception', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--addlayer','-a',action='store_true', help='Add additional layer in fine-tuning')
parser.add_argument('--path', default=cf.test_dir, type=str, help='inference path')
args = parser.parse_args()

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')

cf.test_dir = args.path
data_dir = cf.test_dir
trainset_dir = cf.data_base.split("/")[-1] + os.sep
print("| Preparing %s dataset..." %(cf.test_dir.split("/")[-1]))

use_gpu = torch.cuda.is_available()

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        net = models.alexnet(pretrained=args.finetune)
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        if(args.depth == 16):
            net = models.vgg16(pretrained=args.finetune)
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'inception'):
        net = models.inception_v3(pretrained=args.finetune)
        file_name = 'inceptino-v3'
    elif (args.net_type == 'resnet'):
        net = models.resnet50(args.finetune, args.depth)
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('Error : Network should be either [VGGNet / ResNet]')
        sys.exit(1)

    return net, file_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print("| Loading checkpoint model for inference phase...")
#assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
#assert os.path.isdir('checkpoint/'+trainset_dir), 'Error: No model has been trained on the dataset!'
model, file_name = getNetwork(args)
#checkpoint = torch.load(cf.model_dir)

num_ftrs = model.AuxLogits.fc.in_features
model.AuxLogits.fc = nn.Linear(num_ftrs, cf.num_classes)
# Handle the primary net
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,cf.num_classes)

model.load_state_dict(torch.load(cf.model_dir))

if use_gpu:
    model.cuda()
    cudnn.benchmark = True

model.eval()

sample_input = Variable(torch.randn(1,3,cf.input_size,cf.input_size), volatile=True)
if use_gpu:
    sample_input = sample_input.cuda()

print("\n[Phase 3] : Score Inference")

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

test_transform = transforms.Compose([
    transforms.Scale(cf.input_size),
    transforms.CenterCrop(cf.input_size),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

if not os.path.isdir('result'):
    os.mkdir('result')

output_file = cf.output_dir+cf.test_dir.split("/")[-1]+".csv"

with open(output_file, 'wb') as csvfile:
    fields = ['file_name', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    for subdir, dirs, files in os.walk(data_dir):
        for f in files:
            file_path = subdir + os.sep + f
            if (is_image(f)):
                image = Image.open(file_path).convert('RGB')
                if test_transform is not None:
                    image = test_transform(image)
                inputs = image
                inputs = Variable(inputs, volatile=True)
                if use_gpu:
                    inputs = inputs.cuda()
                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front

                outputs = model(inputs)
                softmax_res = softmax(outputs.data.cpu().numpy()[0])
                score = softmax_res[1]

                print(file_path + "," + str(softmax_res))
                writer.writerow({'file_name': file_path, 'score':softmax_res})
