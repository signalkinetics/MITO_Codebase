"""
This file defines a simple classifier model
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import RandomSampler,WeightedRandomSampler
from tensorboardX import SummaryWriter
import os
import argparse
from sklearn.metrics import confusion_matrix

from model import *



class Net(nn.Module):

    def __init__(self, num_classes, two_channel=False):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(2 if two_channel else 1, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 128, 3, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, 2)
        self.conv4 = nn.Conv2d(256, 512, 3, 2)
        self.conv5 = nn.Conv2d(512, 512, 3, 1)
        self.fc1 = nn.Linear(512 * 11*11, 1024)  
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)


    def forward(self, x):
        

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x))
        return x