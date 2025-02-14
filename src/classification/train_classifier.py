"""
This file is for training classifiers
"""

import numpy as np
import matplotlib.pyplot as plt
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
import sys
import random

sys.path.append('..')
from model import *
from dataloader import mmWaveDataset

# Set seeds for deterministic results
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_model(model, criterion, optimizer, boardio,num_epochs=3,num_classes=5):
    """
    Train a classifier model

    Parameters:
        model: Model object
        criterion: Loss function
        optimizer: Optimizer object
        boardio: Tensorboard object for logging information
        num_epochs: Number of epochs to train for
        num_classes: Number of classes in classifier (used to generate confusion matrix)
    """
    best_val_loss = np.inf
    best_val_nlos_loss = np.inf
    best_train_loss = np.inf
    best_val_los_acc = best_val_nlos_acc = -np.inf
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation_los', 'validation_nlos']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            running_total = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

                # Create confusion matrix every 20 epochs
                if (phase=='validation_los' or phase=='validation_nlos') and epoch%20==0:
                    print(f'Phase: {phase}. ')
                    print(f'Predicted {labels} as {preds}')
                    print(f'Names: {dataloaders[phase].dataset.all_names}')

                    conf_matrix = confusion_matrix(labels.cpu(), preds.cpu(), labels=np.arange(num_classes))

                    print(conf_matrix)

                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_total += inputs.shape[0]
            

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects.float() / running_total

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))
            # Update tensorboard
            if phase =='train':
                best_train_loss = min(best_train_loss, epoch_loss)
                boardio.add_scalar('Train Loss', epoch_loss, epoch+1)
                boardio.add_scalar('Best Train Loss', best_train_loss, epoch+1)
                boardio.add_scalar('Train Accuracy', epoch_acc, epoch+1)
            elif phase =='validation_los':
                if best_val_los_acc <= epoch_acc: torch.save(model.state_dict(), f'checkpoints/{exp_name}/best_los_models/epoch_{epoch}_weights.h5')
                best_val_loss = min(best_val_loss, epoch_loss)
                best_val_los_acc = max(best_val_los_acc, epoch_acc)
                boardio.add_scalar('Val (LOS) Loss', epoch_loss, epoch+1)
                boardio.add_scalar('Best Val (LOS) Loss', best_val_loss, epoch+1)
                boardio.add_scalar('Val (LOS) Accuracy', epoch_acc, epoch+1)
            elif phase =='validation_nlos':
                if best_val_nlos_acc <= epoch_acc: torch.save(model.state_dict(), f'checkpoints/{exp_name}/best_nlos_models/epoch_{epoch}_weights.h5')
                best_val_nlos_loss = min(best_val_nlos_loss, epoch_loss)
                best_val_nlos_acc = max(best_val_nlos_acc, epoch_acc)
                boardio.add_scalar('Val (NLOS) Loss', epoch_loss, epoch+1)
                boardio.add_scalar('Best Val (NLOS) Loss', best_val_nlos_loss, epoch+1)
                boardio.add_scalar('Val (NLOS) Accuracy', epoch_acc, epoch+1)
        # Save models every 100 epochs
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints/{exp_name}/models/epoch_{epoch}_weights.h5')

    return model



if __name__=='__main__':
    print(f'Starting classifer')
    parser = argparse.ArgumentParser(description='mmWave Image Classification')
    parser.add_argument('--exp', type=str, default='tmp', help='name of experiment for tensorboard') 
    parser.add_argument('--plot', type=bool, default=False, help='Plot mmWave images') 
    parser.add_argument('--use_cpu', type=bool, default=False, help='Dont use GPU') 
    parser.add_argument('--cuda_num', type=int, default=0, help='Which GPU number to use') 
    args = parser.parse_args()

    # Parameters for model/training/augmentation/etc
    num_classes = 4 # Number of classes to use
    num_epochs = 500 # Number of epochs
    exp_name = args.exp
    use_cpu = args.use_cpu
    freeze_resnet = True # If using a resnet, whether to freeze the initial layers
    use_simple_model = True # Whether to use a simple model, or a resenet
    num_channels = 1 if use_simple_model else 3 # How many channels to use as input to the model
    use_los_to_train = False # Whether to include LOS images when training the model
    batch_size = 512 # Batch size
    use_full_batch = True # Randomly sample images until full batch is reached
    two_channel = False # Whether to use a two-channel method (where one channel is a binary mask)
    mask_only = False # Whether to use only the mask as input to the classifier
    apply_mask = True # Whether to apply a mask (output from SAM) to the image 
    use_diff=False # Whether to use the diffuse simulation in training
    use_spec=True # Whether to use the specular simulation in training
    use_edge=True # Whether to use the edge simulaton in training
    gaussian_blur=True # Whether to apply gaussian blur to images 
    add_noise = True # Whether to add nosie when training
    large_fc = True # Whether to use a larger fully connected layer
    conditional_noise = True # If True, only apply noise on some images during training
    dilate_mask = False # Whether to dilate masks before applying them
    use_new_mask = False # Whether to use the newer masks
    use_multi_mask=False # Whether to use many different versions of a mask 
    relative_norm = True # Whether to normalize each image relative to itself
    use_new_data = True # Whether to use newest round of data
    clip_vals = True # Whether to clip values that are too large
    fixed_exp_num = None # None to use all experiments, or '1', or '2' to use only data from experiments 1 or 2

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


    boardio = SummaryWriter(log_dir='checkpoints/' + exp_name)
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'best_nlos_models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'best_nlos_models')
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'best_los_models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'best_los_models')

    # Create datasets
    image_datasets = {
        'train':           mmWaveDataset(phase='train', plot=False, **augmentation_parameters),
        'validation_los':  mmWaveDataset(phase='validation', augment=False, plot=args.plot, los_only='los', **augmentation_parameters),
        'validation_nlos': mmWaveDataset(phase='validation', augment=False, plot=args.plot, los_only='nlos', **augmentation_parameters),
    }

    # Create dataloaders
    samples_weight = image_datasets['train'].get_sample_weights()
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler_train = WeightedRandomSampler(samples_weight, replacement=True, num_samples=batch_size)
    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=batch_size, sampler=sampler_train if use_full_batch else None,
                                    shuffle=False, num_workers=4),
        'validation_los':
        torch.utils.data.DataLoader(image_datasets['validation_los'],
                                    batch_size=len(image_datasets['validation_los']), sampler= None,
                                    shuffle=False, num_workers=4),

        'validation_nlos':
        torch.utils.data.DataLoader(image_datasets['validation_nlos'],
                                    batch_size=len(image_datasets['validation_nlos']), sampler= None,
                                    shuffle=False, num_workers=4)
    }
    print(f'Have {len(image_datasets["validation_los"])} LOS points')
    print(f'Have {len(image_datasets["validation_nlos"])} NLOS points')

    # Select a device (CPU or GPU)
    if use_cpu:
        device = torch.device("cpu")
    else:
        print(f'Using device {f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"}')
        device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    # Create model
    if use_simple_model:
        model = Net(num_classes, two_channel).to(device)
    else:
        model = models.resnet50(pretrained=True).to(device)
        
        if freeze_resnet:
            for param in model.parameters():
                param.requires_grad = False   
            
        if large_fc:
            model.fc = nn.Sequential(
                        nn.Linear(2048, 2048),
                        nn.ReLU(inplace=True),
                        nn.Linear(2048, 1024),
                        nn.ReLU(inplace=True),
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, num_classes)).to(device)

        else:
            model.fc = nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, num_classes)).to(device)

    # Create loss/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train model
    model_trained = train_model(model, criterion, optimizer,boardio, num_epochs=num_epochs,num_classes=num_classes)

    # Save final model
    torch.save(model_trained.state_dict(), f'checkpoints/{exp_name}/models/final_weights.h5')

