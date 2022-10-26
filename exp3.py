# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:49:48 2022

@author: user
"""


import os
import torch
import argparse
import numpy as np
from models import get_model, ResNet20, Vgg16, DenseNet
from matplotlib import pyplot as plt
from dataset import get_dataloaders
from attacks_transfer import test_FGSM_new, test_IFGSM_new, test_MIFGSM_new

def get_attack_results(model, model_attack, test_dl, epsilon_list, device):
    FGSM_accs = []
    print('performing attack with FGSM\n')
    for epsilon in epsilon_list:
        acc = test_FGSM_new(model, model_attack, test_dl, epsilon, device)
        FGSM_accs.append(acc)
    
    torch.cuda.empty_cache()
    
    IFGSM_accs = []
    print('performing attack with IFGSM\n')
    for epsilon in epsilon_list:
        acc = test_IFGSM_new(model, model_attack, test_dl, epsilon, device)
        IFGSM_accs.append(acc)
        
    torch.cuda.empty_cache()
    
    MIFGSM_accs = []
    print('performing attack with MIFGSM\n')
    for epsilon in epsilon_list:
        acc = test_MIFGSM_new(model, model_attack, test_dl, epsilon, device)
        MIFGSM_accs.append(acc)
    
    torch.cuda.empty_cache()
    
    return FGSM_accs, IFGSM_accs, MIFGSM_accs


def get_plots(model_name):
    plt.figure()
    plt.plot(epsilon_list, FGSM_accs, label='FGSM')
    plt.plot(epsilon_list, IFGSM_accs, label='IFGSM')
    plt.plot(epsilon_list, MIFGSM_accs, label='MIFGSM')
    plt.xlabel('epsilon')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.title(f'Transfer Attacks from {model_name} on {args.dataset}')
    
    os.makedirs(f'plots/{args.dataset}', exist_ok=True)
    
    plt.savefig(f'plots/{args.dataset}/exp3_{model_name}.PNG')
    
    
#%%   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset for performing attacks')
    args = parser.parse_args()
    
    weight_path = f'saved_models/vanilla_{args.dataset}.pth'
    
    if args.dataset == 'mnist':
        epsilon_list = [0,0.02,0.04,0.06,0.08,0.1]
    else:
        epsilon_list = [0,0.002,0.004,0.006,0.008,0.01]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model('vanilla', args.dataset)
    model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    _, _, test_dl = get_dataloaders(64, dataset=args.dataset)
    
    
    for Model, model_name in zip([ResNet20, Vgg16, DenseNet], ['resnet20', 'vgg16', 'densenet121']):
        
        print(f'Performing transfer attack with {model_name}\n')
        
        # dict_ = np.load(f'attack_results/{args.dataset}/exp2_{model_name}.npy', allow_pickle=True).item()
        # FGSM_accs = dict_['FGSM']
        # IFGSM_accs = dict_['IFGSM']
        # MIFGSM_accs = dict_['MIFGSM']
    
        model_attack = Model(args.dataset)
        model_attack.to(device)
        model_attack.load_state_dict(torch.load(f'saved_models/{model_name}_{args.dataset}.pth'))
        model_attack.eval()
    
        FGSM_accs, IFGSM_accs, MIFGSM_accs = get_attack_results(model, model_attack, test_dl, 
                                                                epsilon_list, device)
        
        os.makedirs(f'attack_results/{args.dataset}', exist_ok=True)
        np.save(f'attack_results/{args.dataset}/exp3_{model_name}.npy', 
                {'FGSM': FGSM_accs, 'IFGSM': IFGSM_accs,
                  'MIFGSM': MIFGSM_accs})
    
        get_plots(model_name)