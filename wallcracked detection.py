# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:00:30 2022

@author: HP
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('D:\Bharti\crack-identificationwall'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
# define training and test data directories
train_dir =('D:\\Bharti\\crack-identificationwall\\train')
test_dir = ('D:\\Bharti\\crack-identificationwall\\test')

# classes are folders in each directory with these names
classes = ['cracked','uncracked']

# load and transform data using ImageFolder

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder('D:\\Bharti\\crack-identificationwall\\train', transform=data_transform)
#test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
#print('Num test images: ', len(test_data))


# define dataloader parameters
batch_size = 32
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                        #  num_workers=num_workers, shuffle=True)
                                        
# Visualize some sample data
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

# print out the model structurefig = plt.figure(figsize=(25, 4))
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
    
print(vgg16)

print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features)

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

    
import torch.nn as nn
n_inputs = vgg16.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = true
last_layer = nn.Linear(n_inputs, len(classes))

vgg16.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16.cuda()

# check to see that your last layer produces the expected number of outputs
print(vgg16.classifier[6].out_features)
#print(vgg16)

import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.005)

n_epochs = 30

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
data_dir = 'crack-detection-ce784a-iitk/'
test_dir = os.path.join(data_dir, 'test/')
    
    

















                                        
                                        
                                        
                                        
                                        
                                        
                                        
