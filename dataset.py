# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:08:08 2022

@author: user
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class AdvDataset(Dataset):
    def __init__(self, dataset, mode='train'):
        # if dataset =='mnist':
        #     orig_x = np.load('data/original_x.npy')
        #     adv_x = np.load('data/adversarial_x.npy')
        #     label = np.load('data/original_y.npy')
        # else:
        
        orig_x = np.load(f'adv_examples/{dataset}/original_x_{mode}.npy')
        adv_x = np.load(f'adv_examples/{dataset}/adversarial_x_{mode}.npy')
        label = np.load(f'adv_examples/{dataset}/original_y_{mode}.npy')
            
        self.xs = np.concatenate((orig_x, adv_x), axis=0)
        self.ys = np.concatenate((label, label), axis=0)
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        
        x = torch.from_numpy(x)
        y = torch.tensor(y)
        return x, y  
  
def get_dataloaders(batch_size, dataset='cifar10', retraining=False):
    transform = transforms.Compose([transforms.ToTensor()]) 
    
    if dataset == 'mnist':
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    else:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])
    
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

        test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        

    train_indices, val_indices, _, _ = train_test_split(range(len(train_ds)),
                                                        train_ds.targets,
                                                        stratify=train_ds.targets,
                                                        test_size=0.1,
                                                        random_state=0)

    if retraining:
        train_ds = AdvDataset(dataset, mode='train')
        valid_ds = AdvDataset(dataset, mode='valid')
        
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        # generate subset based on indices
        train_split = Subset(train_ds, train_indices)
        val_split = Subset(train_ds, val_indices)

        train_dl = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_dl = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
    return train_dl, valid_dl, test_dl

if __name__ == '__main__':
    train_dl, valid_dl, test_dl = get_dataloaders(64)