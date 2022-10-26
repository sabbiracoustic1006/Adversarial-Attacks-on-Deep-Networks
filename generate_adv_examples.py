# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:36:29 2022

@author: user
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from models import get_model
import torch.nn.functional as F
from dataset import get_dataloaders

def adv_data_collector(net, testloader):

    correct = 0
    listx_1 = []
    listx_2 = []
    listy = []
    listy_2 = []
    
    if args.dataset == 'mnist':
        epsilon_list = [0.02,0.04,0.06,0.08,0.1]
    else:
        epsilon_list = [0,0.002,0.004,0.006,0.008,0.01]

    for image, label in tqdm(testloader):

        image, label = image.cuda(), label.cuda()

        image.requires_grad = True
        
        img_ls1 = image.clone().detach().cpu().numpy()
        label_ls1 = label.detach().cpu().numpy()
        listx_1.append(img_ls1)
        listy.append(label_ls1)

        output = net(image)
       
        pred = torch.argmax(output, dim =1) 

        loss = F.cross_entropy(output, label)

        net.zero_grad()

        loss.backward()

        data_grad = image.grad.data

        
        p = np.random.randint(len(epsilon_list))
        epsilon = epsilon_list[p]

        perturbed_image = image + epsilon*(data_grad.sign())
        ## save here
        img_ls2 = perturbed_image.clone().detach().cpu().numpy()
        listx_2.append(img_ls2)

        output = net(perturbed_image)

        final_pred = torch.argmax(output, dim =1) 

        listy_2.append(final_pred.detach().cpu().numpy())

        if final_pred.item() == label.item():
            correct += 1

    final_acc = correct/float(len(testloader))
    print("Epsilon: {}, Test Accuracy = {}".format(epsilon, final_acc))

    return listx_1, listx_2, listy, listy_2

def save_adv_examples(model, train_dl, valid_dl, args):
    os.makedirs(f'adv_examples/{args.dataset}', exist_ok=True)
    
    listx1, listx2, listy1, listy2 = adv_data_collector(model, train_dl)

    arr_x1 = np.concatenate(listx1,axis=0)
    np.save(f'adv_examples/{args.dataset}/original_x_train.npy', arr_x1)
    
    arr_x2 = np.concatenate(listx2,axis=0)
    np.save(f'adv_examples/{args.dataset}/adversarial_x_train.npy', arr_x2)
    
    arr_y1 = np.concatenate(listy1,axis=0)
    np.save(f'adv_examples/{args.dataset}/original_y_train.npy', arr_y1)
    
    listx1, listx2, listy1, listy2 = adv_data_collector(model, valid_dl)
    
    arr_x1 = np.concatenate(listx1,axis=0)
    np.save(f'adv_examples/{args.dataset}/original_x_valid.npy', arr_x1)
    
    arr_x2 = np.concatenate(listx2,axis=0)
    np.save(f'adv_examples/{args.dataset}/adversarial_x_valid.npy', arr_x2)
    
    arr_y1 = np.concatenate(listy1,axis=0)
    np.save(f'adv_examples/{args.dataset}/original_y_valid.npy', arr_y1)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset for performing attacks')
    args = parser.parse_args()
    
    weight_path = f'saved_models/vanilla_{args.dataset}.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model('vanilla', args.dataset)
    model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    train_dl, valid_dl, _ = get_dataloaders(1, dataset=args.dataset)
    
    save_adv_examples(model, train_dl, valid_dl, args)
    
    