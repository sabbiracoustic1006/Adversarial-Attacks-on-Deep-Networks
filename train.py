# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:31:30 2022

@author: user
"""


import os
import torch
import argparse
from torch import nn
from models import get_model
from tqdm import tqdm
from dataset import get_dataloaders
# from attacks import test_FGSM_new
from utils import SEED_EVERYTHING, evaluate, AverageMeter, accuracy_score

def train(args):
    
    lr = args.lr
    n_epochs = args.n_epochs 
    batch_size = args.batch_size
        
    SEED_EVERYTHING()

    train_dl, valid_dl, test_dl = get_dataloaders(batch_size, args.dataset)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = get_model(args.model, args.dataset)
    model.to(device)
    
    os.makedirs('saved_models', exist_ok=True)
    
    # if load_pretrained_weight:
    #     model.load_state_dict(torch.load('saved_models/vanilla_net.pth'))
    #     print('pretrained weight loaded')
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0
    
    # train_losses = []
    # train_accs = []
    # test_losses = []
    # test_accs = []
    
    for epoch_i in range(n_epochs):         
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(train_dl, total=len(train_dl))
        model.train() 
        
        for step, batch in enumerate(pbar):
            img = batch[0].to(device)
            tgt = batch[1].to(device)
                  
            pred = model(img)
            loss = loss_func(pred, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred_label = pred.argmax(1).cpu().numpy()
            tgt_label = batch[1].numpy()
            
            acc = accuracy_score(pred_label, tgt_label)
            
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            
            pbar.set_postfix({'loss':loss_meter.avg, 'accuracy':acc_meter.avg})
        
        valid_loss, valid_acc = evaluate(model, valid_dl, device, loss_func)   
        test_loss, test_acc = evaluate(model, test_dl, device, loss_func)   
    
        # train_losses.append(loss_meter.avg)
        # train_accs.append(acc_meter.avg)
      
        # test_losses.append(test_loss)
        # test_accs.append(test_acc)
        print('test accuracy:', test_acc)
        if valid_acc > best_acc:
            torch.save(model.state_dict(), f'saved_models/{args.model}_{args.dataset}.pth')
            print('Validation accuracy improved from %.4f to %.4f'%(best_acc,valid_acc))
            best_acc = valid_acc
      
        print(f'Epoch: {epoch_i+1}/{n_epochs}, train loss:{loss_meter.avg:.4f}, train accuracy:{acc_meter.avg:.4f}\nvalid loss:{valid_loss:.4f} valid accuracy: {valid_acc:.4f}')
    # return train_losses, train_accs, test_losses, test_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate to use for training')
    parser.add_argument('--n_epochs', default=50, type=int, help='batch size to be used for training')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size to be used for training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='choose between mnist and cifar100 dataset')
    parser.add_argument('--model', default='vanilla', type=str, help='choose between vanilla, resnet20, vgg16, densenet121')
    args = parser.parse_args()
    
    train(args)