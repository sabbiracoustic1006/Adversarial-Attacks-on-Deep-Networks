# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:02:14 2022

@author: user
"""

import os
import torch
import argparse
import numpy as np
from models import get_model
from matplotlib import pyplot as plt
from dataset import get_dataloaders
from attacks_mod import test_FGSM_new, test_IFGSM_new, test_MIFGSM_new

def get_attack_results(model, test_dl, epsilon_list):
    FGSM_accs = []
    # print('performing attack with FGSM\n')
    # for epsilon in epsilon_list:
    #     acc = test_FGSM_new(model, test_dl, epsilon)
    #     FGSM_accs.append(acc)
        
    # torch.cuda.empty_cache()
    
    IFGSM_accs = []
    # print('performing attack with IFGSM\n')
    # for epsilon in epsilon_list:
    #     acc = test_IFGSM_new(model, test_dl, epsilon)
    #     IFGSM_accs.append(acc)
    
    # torch.cuda.empty_cache()
    
    MIFGSM_accs = []
    print('performing attack with MIFGSM\n')
    for epsilon in epsilon_list:
        acc = test_MIFGSM_new(model, test_dl, epsilon)
        MIFGSM_accs.append(acc)
        
    torch.cuda.empty_cache()
    
    return FGSM_accs, IFGSM_accs, MIFGSM_accs


def get_plots():
    plt.plot(epsilon_list, MIFGSM_accs_before, label='MIFGSM')
    plt.plot(epsilon_list, MIFGSM_accs, label='MIFGSM (perceptually guided)')
    plt.xlabel('epsilon')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.title(f'Perceptually guided attack on {args.dataset}')
    
    os.makedirs(f'plots/{args.dataset}', exist_ok=True)
    
    plt.savefig(f'plots/{args.dataset}/exp4.PNG')
    
    
#%%   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset for performing attacks')
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
    
    FGSM_accs, IFGSM_accs, MIFGSM_accs = get_attack_results(model, test_dl, epsilon_list)
    
    dict_ = np.load(f'attack_results/{args.dataset}/exp1.npy', allow_pickle=True).item()
    MIFGSM_accs_before = dict_['MIFGSM']
    
    os.makedirs(f'attack_results/{args.dataset}', exist_ok=True)
    np.save(f'attack_results/{args.dataset}/exp4.npy', {'FGSM': FGSM_accs, 'IFGSM': IFGSM_accs,
                                                        'MIFGSM': MIFGSM_accs})
    
    get_plots()