# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:22:59 2022

@author: user
"""

import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from hessian import hessian

def get_FGSM(images, labels, model, eps):
    images.requires_grad = True
    
    pred = model(images)
    loss = F.cross_entropy(pred, labels)    
    
    model.zero_grad()
    loss.backward()
    G = images.grad.data
    
    perturbed_image = images + eps*(G.sign())
    return perturbed_image.detach()

def test_FGSM_new(model, test_dl, epsilon):
    correct = 0
    
    model.eval()
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.cuda(), labels.cuda()
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_FGSM(images, labels, model, epsilon)
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc


def get_IFGSM(images, labels, model, eps, iters=10) :

    clamp_max = 1.0
    alpha = eps / iters
              
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        attack_images = images + alpha*images.grad.sign()
                
        # a = max{0, X-eps}
        a = torch.clamp(images - eps, min=0)
        # b = max{a, X'}
        b = (attack_images>=a).float()*attack_images + (a>attack_images).float()*a
        # c = min{X+eps, b}
        c = (b > images+eps).float()*(images+eps) + (images+eps >= b).float()*b
        # d = min{255, c}
        images = torch.clamp(c, max=clamp_max).detach()
            
    return images

def test_IFGSM_new(model, test_dl, epsilon):
    correct = 0
    
    model.eval()
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.cuda(), labels.cuda()
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_IFGSM(images, labels, model, epsilon)
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc



def get_MIFGSM_new(image,label,net,epsilon,num_iter=10,mu = 1):

  alpha = epsilon/num_iter

  g = 0
  # image.requires_grad = True
  x = image.clone()
  # x.requires_grad = True

  for itr in range(num_iter):
      
    x.requires_grad = True

    y = net(x)

    loss = F.cross_entropy(y, label)

    net.zero_grad()

    loss.backward()

    data_grad = x.grad.data
    
    # data_hess = hessian(loss, x, create_graph=True)
    
    # print(data_hess.shape)
    # print(x)

    g = mu*g + (data_grad)/data_grad.norm(p=1, dim=(1,2,3), keepdim=True)

    x = x + alpha*(g.sign())#).detach()
    
    x = x.detach()

  return x


def test_MIFGSM_new(model, test_dl, epsilon):
    correct = 0
    
    model.eval()
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.cuda(), labels.cuda()
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_MIFGSM_new(images, labels, model, epsilon)
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc

