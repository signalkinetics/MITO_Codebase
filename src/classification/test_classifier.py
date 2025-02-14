"""
This file tests a classifier after training
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler,WeightedRandomSampler
import os
import argparse
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize
import sys

sys.path.append('..')
from model import *
from dataloader import mmWaveDataset

def test_model(model, num_classes=5):
    """
    Test a trained model
    """
    best_val_loss = np.inf
    best_val_nlos_loss = np.inf
    best_train_loss = np.inf
    for phase in ['test_all', 'test_los', 'test_nlos']: # Test all, LOS only, and NLOS only
        model.eval()

        running_loss = 0.0
        running_corrects = 0.0
        running_total = 0.0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            print(f'Phase: {phase}. ')
            print(f'Predicted {labels} as {preds}')
            print(f'Names: {dataloaders[phase].dataset.all_names}')

            # Create confusion matrix
            conf_matrix = confusion_matrix(labels.cpu(), preds.cpu(), labels=np.arange(num_classes))
            print(conf_matrix)
            conf_matrix = conf_matrix.astype(np.float32)
            print(conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis])

            running_corrects += torch.sum(preds == labels.data)
            running_total += inputs.shape[0]
        
        epoch_acc = running_corrects.float() / running_total

        print('{} acc: {:.4f}'.format(phase,
                                                    epoch_acc.item()))
            

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='mmWave Image Classification')
    parser.add_argument('--exp', type=str, default='tmp', help='name of experiment for tensorboard') 
    parser.add_argument('--plot', type=bool, default=False, help='Plot mmWave images') 

    parser.add_argument('--use_cpu', type=bool, default=False, help='Dont use GPU') 
    parser.add_argument('--cuda_num', type=int, default=0, help='Dont use GPU') 
    args = parser.parse_args()

    # Parameters for model. This must match those used to train the model. See train_classifier for more details on this
    num_classes = 4
    num_epochs = 5000
    exp_name = args.exp
    use_cpu = args.use_cpu
    freeze_resnet = True
    use_simple_model = True#False 
    num_channels = 1 if use_simple_model else 3
    use_los_to_train = False
    batch_size = 512
    batch_size_test = 32
    use_full_batch = True
    two_channel = False
    mask_only = False
    apply_mask = True
    use_diff=False
    use_spec=True
    use_edge=True
    gaussian_blur=True
    add_noise = False
    large_fc = True
    conditional_noise = True
    dilate_mask = False
    use_new_mask = False
    use_multi_mask=False
    relative_norm = True
    clip_vals = True
    use_new_data = True
    fixed_exp_num = None

    augmentation_parameters = {
        'add_noise': add_noise, 
        'conditional_noise': conditional_noise, 
        'dilate_mask': dilate_mask, 
        'use_new_mask': use_new_mask, 
        'use_multi_mask': use_multi_mask, 
        'relative_norm': relative_norm, 
        'apply_mask': apply_mask, 
        'two_channel': two_channel, 
        'num_channels': num_channels, 
        'mask_only': mask_only,
        'gaussian_blur': gaussian_blur,  
        'use_diff': use_diff,
        'use_spec': use_spec,
        'use_edge': use_edge,
        'use_new_data': use_new_data,
        'fixed_exp_num': fixed_exp_num,
        'clip_vals': clip_vals
    }

    # Path of trained model. These are the paths to the models used in the paper. To download pretrained weights, please see the README
    path = f'checkpoints/1103c_final_all/models/final_weights.h5' # Full classifier
    # path = f'checkpoints/1103c_final_edge/models/final_weights.h5' # Microbenchmark: Training with only edge simulation
    # path = f'checkpoints/1103c_final_spec/models/final_weights.h5' # Microbenchmark: Training with only specular simulation

    boardio = SummaryWriter(log_dir='checkpoints/' + exp_name)
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'models')

    # Create datasets
    image_datasets = {
        'test_all': mmWaveDataset(phase='test', augment=False, plot=args.plot, **augmentation_parameters),
        'test_los': mmWaveDataset(phase='test', augment=False, plot=args.plot, los_only='los', **augmentation_parameters),
        'test_nlos': mmWaveDataset(phase='test',  augment=False, plot=args.plot, los_only='nlos', **augmentation_parameters),
    }

    # Create dataloaders
    dataloaders = {
        'test_all':
        torch.utils.data.DataLoader(image_datasets['test_all'],
                                    batch_size=len(image_datasets['test_all']), sampler= None,
                                    shuffle=False, num_workers=4),
        'test_los':
        torch.utils.data.DataLoader(image_datasets['test_los'],
                                    batch_size=len(image_datasets['test_los']), sampler= None,
                                    shuffle=False, num_workers=4),

        'test_nlos':
        torch.utils.data.DataLoader(image_datasets['test_nlos'],
                                    batch_size=len(image_datasets['test_nlos']), sampler= None,
                                    shuffle=False, num_workers=4)
    }

    print(f'Have {len(image_datasets["test_all"])} all points')
    print(f'Have {len(image_datasets["test_los"])} NLOS points')
    print(f'Have {len(image_datasets["test_nlos"])} NLOS points')
    if use_cpu:
        device = torch.device("cpu")
    else:
        print(f'Using device {f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"}')
        device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    
    # Load and test model
    model = Net(num_classes, two_channel).to(device)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    test_model(model,num_classes=num_classes)

