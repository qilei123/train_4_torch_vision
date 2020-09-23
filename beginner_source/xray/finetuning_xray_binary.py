# -*- coding: utf-8 -*-
"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
#sys.path.insert(0,"../")
from FocalLoss import FocalLoss
#sys.path.insert(0,'/data0/qilei_chen/pytorch_vision_4_DR')
sys.path.insert(0,'/data1/qilei_chen/DEVELOPMENTS/vision2')
#sys.path.insert(0,"/data1/qilei_chen/DEVELOPMENTS/vision")
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image,ImageOps
import random
print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

import argparse
import cv2
def pil_loader(path):
    '''
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    '''
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(img).convert('RGB')

class XrayDataset(VisionDataset):
    def __init__(self, root, file_names,labels,transforms=None, transform=None, target_transform=None,is_shuffle_sample=False):
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.root = root
        self.file_names = file_names
        self.labels = labels
        self.transforms = transforms
        self.target_transform = target_transform
        self.is_shuffle_sample = is_shuffle_sample

        if self.is_shuffle_sample:
            max_count_category = 0
            max_count = 0
            labels_instances = dict()

            for label,index in zip(self.labels,range(len(self.labels))):
                if label in labels_instances:
                    pass
                else:
                    labels_instances[label] = []
                labels_instances[label].append(index)
                if len(labels_instances[label])>max_count:
                    max_count_category=label
                    max_count = len(labels_instances[label])
            labels_instances_add = dict()
            for label in labels_instances:
                more_count = max_count-len(labels_instances[label])
                labels_instances_add[label]=[]
                for i in range(more_count):
                    randint = random.randint(0,len(labels_instances[label])-1)
                    labels_instances_add[label].append(labels_instances[label][randint])
            balanced_indexes = []

            for label in labels_instances:
                balanced_indexes+=labels_instances[label]
                balanced_indexes+=labels_instances_add[label]

            random.shuffle(balanced_indexes)
            balanced_labels = []
            balanced_file_names = []
            for i in range(len(balanced_indexes)):
                balanced_labels.append(labels[balanced_indexes[i]])
                balanced_file_names.append(file_names[balanced_indexes[i]])
            
            self.labels = balanced_labels
            self.file_names = balanced_file_names
            

    def __getitem__(self, index):
        image_dir = os.path.join(self.root,self.file_names[index])
        image = pil_loader(image_dir)

        if self.transform is not None:
            image = self.transform(image)
        if self.transforms is not None:
            image = self.transforms(image)
        target = self.labels[index]
        if self. target_transform is not None:
            target = self.target_transform(target)
        return image,target

    def __len__(self):
        return len(self.labels)

parser = argparse.ArgumentParser(description='model name')
parser.add_argument('--model', '-m', help='set the training model', default="alexnet")
parser.add_argument('--datadir', '-d', help='set the training dataset', default="/data2/qilei_chen/DATA/xray")
parser.add_argument('--imagefolder', '-i', help='the folder for the images', default="xray_images1")
parser.add_argument('--annotation', '-a', help='the annotations for the images', default="xray_dataset_simple_annotations.csv")
parser.add_argument('--kcross', '-k', help='set the number of k cross folder', default=4)
parser.add_argument('--classnumber', '-c', help='set the classes of label', default=2)
args = parser.parse_args()


######################################################################
# Inputs
# ------
# 
# Here are all of the parameters to change for the run. We will use the
# *hymenoptera_data* dataset which can be downloaded
# `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__.
# This dataset contains two classes, **bees** and **ants**, and is
# structured such that we can use the
# `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__
# dataset, rather than writing our own custom dataset. Download the data
# and set the ``data_dir`` input to the root directory of the dataset. The
# ``model_name`` input is the name of the model you wish to use and must
# be selected from this list:
# 
# ::
# 
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
# 
# The other inputs are as follows: ``num_classes`` is the number of
# classes in the dataset, ``batch_size`` is the batch size used for
# training and may be adjusted according to the capability of your
# machine, ``num_epochs`` is the number of training epochs we want to run,
# and ``feature_extract`` is a boolean that defines if we are finetuning
# or feature extracting. If ``feature_extract = False``, the model is
# finetuned and all model parameters are updated. If
# ``feature_extract = True``, only the last layer parameters are updated,
# the others remain fixed.
# 

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
#data_dir = "/data2/DB_GI/0/sample2"

data_dir = args.datadir
image_folder = args.imagefolder
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"
model_name = "resnet"
model_name = "alexnet"
model_name = "vgg"
model_name = "densenet"
model_name = "inception"
model_name = args.model
print("-------------------"+model_name+"-------------------")

#model_folder_dir = data_dir+'/balanced_finetune_4_'+model_name
model_folder_dir = data_dir+'/balanced_finetune_2_'+model_name
if not os.path.exists(model_folder_dir):
    os.makedirs(model_folder_dir)

# Number of classes in the dataset
num_classes = args.classnumber

# Batch size for training (change depending on how much memory you have)
if model_name=="vgg" or model_name=="resnet101":
    batch_size = 4
else:
    batch_size = 16

# Number of epochs to train for 
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

input_size_ = 299

gpu_index = '0'
device = torch.device("cuda:"+gpu_index)# if torch.cuda.is_available() else "cpu")
######################################################################
# Helper Functions
# ----------------
# 
# Before we write the code for adjusting the models, lets define a few
# helper functions.
# 
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
# 

def train_model(model, dataloaders, criterion, optimizer, folderindex,num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        model_save_path = model_folder_dir+'/'+model_name+'_epoch_'+str(epoch)+'.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_path)  
        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_folder_dir+'/'+str(folderindex)+'_best.model')
    return model, val_acc_history


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
# 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext50_32x4d":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224   

    elif model_name == "resnext101_32x8d":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224         

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11":

        model_ft = models.vgg11(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg13":

        model_ft = models.vgg13(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16":

        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":

        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":

        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg13_bn":

        model_ft = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16_bn":

        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19_bn":

        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet1_0":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "squeezenet1_1":
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "densenet161":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "densenet201":
        """ Densenet
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "shufflenetv2_x0_5":
        """ Densenet
        """
        model_ft = models.shufflenetv2_x0_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "shufflenetv2_x1_0":
        """ Densenet
        """
        model_ft = models.shufflenetv2_x1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "shufflenetv2_x1_5":
        """ Densenet
        """
        model_ft = models.shufflenetv2_x1_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "shufflenetv2_x2_0":
        """ Densenet
        """
        model_ft = models.shufflenetv2_x2_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "mobilenet_v2":
        """ Densenet
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "inception3":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = input_size_

    elif model_name == "inceptionv4":
        """ Inception v4
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inceptionv4(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = input_size_

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size



import pandas as pd
import numpy as np
from  sklearn.model_selection import KFold
LABEL_MAP = ['0','1']

def split_set(csv, nsplit = 4):

    pd_frame = pd.read_csv(csv, sep=',')

    file_name = pd_frame.filename.to_numpy()
    label = pd_frame.abnormal.to_numpy()
    # remove NaN rows
    #keep = label != 'nan'
    #file_name = file_name[keep]
    #label = label[keep]
    # convert string label to numeric label
    assert len(np.unique(label)) == len(LABEL_MAP)
    for i, l in enumerate(LABEL_MAP):
        label[label == l] = i
    label = label.astype(np.int)
    unique, counts = np.unique(label, return_counts=True)
    print(np.asarray((unique, counts)).T)
    # build 4 fold
    # random seed for reproducibility
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=20)
    for train_index, test_index in kf.split(file_name):
        yield file_name[train_index], label[train_index],file_name[test_index], label[test_index]


def cross_validation():
    avg_precision = 0
    print(model_name)
    precision_records = open(os.path.join(data_dir,"records.txt"),"w")
    for i,(train_file_names,train_labels,val_file_names,val_labels) in enumerate(split_set(
        os.path.join(args.datadir,args.annotation),nsplit=args.kcross)):
        print("-------Round "+str(i)+"-------")
        print("number of training images:"+str(len(train_labels)))
        print("number of validation images:"+str(len(val_labels)))
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        # Print the model we just instantiated
        #print(model_ft) 

        print("Initializing Datasets and Dataloaders...")
        ######################################################################
        # Load Data
        # ---------
        # 
        # Now that we know what the input size must be, we can initialize the data
        # transforms, image datasets, and the dataloaders. Notice, the models were
        # pretrained with the hard-coded normalization values, as described
        # `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
        # 

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(input_size),
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.RandomRotation2(),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # Create training and validation datasets
        #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        image_root = os.path.join(data_dir,image_folder)
        image_datasets = {'train':XrayDataset(image_root,train_file_names,train_labels,data_transforms["train"],is_shuffle_sample=False),
            'val':XrayDataset(image_root,val_file_names,val_labels,data_transforms["val"])}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Detect if we have a GPU available

        #device = torch.device('cpu')

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are 
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        #print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    #print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    #print("\t",name)
                    pass

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)


        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        #criterion = FocalLoss(class_num = num_classes,device_index=int(gpu_index))

        # Train and evaluate
        model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,i, num_epochs=num_epochs, is_inception=(model_name=="inception"))


        ohist = []

        ohist = [h.cpu().numpy() for h in hist]
        for precision in ohist:
            precision_records.write(str(precision))
            precision_records.write(" ")
        precision_records.write("\n")
            
        avg_precision += max(ohist) 
    print(str(args.kcross)+"-crocss folder average precision:"+str(avg_precision/args.kcross))
if __name__ == "__main__":
    cross_validation()    