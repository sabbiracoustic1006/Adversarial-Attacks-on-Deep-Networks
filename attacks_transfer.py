# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:22:59 2022

@author: user
"""

import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F

def get_FGSM(images, labels, model, eps):
    images.requires_grad = True
    
    pred = model(images)
    loss = F.cross_entropy(pred, labels)    
    
    model.zero_grad()
    loss.backward()
    G = images.grad.data
    
    perturbed_image = images + eps*(G.sign())
    # print(perturbed_image.norm(p=2))
    return perturbed_image.detach()

def test_FGSM_new(model, model_attack, test_dl, epsilon, device):
    correct = 0
    
    model.eval()
    model_attack.eval()
    
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_FGSM(images, labels, model_attack, epsilon)
        
        # return 0
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc

def test_FGSM(net, testloader, epsilon, device):

    correct = 0

    net = net.eval()

    # criterion = nn.CrossEntropyLoss()

    for image, label in tqdm(testloader,total=len(testloader)):

        image, label = image.to(device), label.to(device)

        image.requires_grad = True

        output = net(image)
       
        pred = torch.argmax(output, dim =1) 

        if pred.item() != label.item():
          continue

        loss = F.cross_entropy(output, label)

        net.zero_grad()

        loss.backward()

        data_grad = image.grad.data

        perturbed_image = image + epsilon*(data_grad.sign())
        ## save here

        output = net(perturbed_image)

        final_pred = torch.argmax(output, dim =1) 

        if final_pred.item() == label.item():
            correct += 1

    final_acc = correct/float(len(testloader))
    print("Epsilon: {}, Test Accuracy = {}".format(epsilon, final_acc))

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

def test_IFGSM_new(model, model_attack, test_dl, epsilon, device):
    # return 0
    correct = 0
    
    model.eval()
    model_attack.eval()
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_IFGSM(images, labels, model_attack, epsilon)
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc

def test_IFGSM(net, device, testloader, epsilon):

    correct = 0
    pbar = tqdm(testloader, total=len(testloader))
    net = net.eval()

    for step, (image, label) in enumerate(pbar, 1):

        image, label = image.to(device), label.to(device)

        image.requires_grad = True

        output = net(image)
       
        pred = torch.argmax(output)

        if pred.item() != label.item():
          continue

        perturbed_image = get_IFGSM(image,label,net,epsilon)

        output = net(perturbed_image)

        final_pred = torch.argmax(output)

        if final_pred.item() == label.item():
            correct += 1

        pbar.set_postfix({'acc': correct / step})

    final_acc = correct/float(len(testloader))
    print("Epsilon: {}, Test Accuracy = {}".format(epsilon, final_acc))

    return final_acc



def get_MIFGSM(image,label,net,epsilon,num_iter,mu = 1):

  alpha = epsilon/num_iter

  g = 0

  x = image

  for itr in range(num_iter):

    y = net(x)

    loss = F.cross_entropy(y, label)

    net.zero_grad()

    loss.backward(retain_graph=True)

    data_grad = image.grad.data
    
    print('data_grad', data_grad.shape)

    g = mu*g + (data_grad)/torch.norm(data_grad, p = 1)
    
    print('norm', torch.norm(data_grad, p = 1))
    print('g', g.shape)
    x = x + alpha*(g.sign()) 

  return x

def get_MIFGSM_new(image,label,net,epsilon,num_iter=10,mu = 1):

  alpha = epsilon/num_iter

  g = 0
  
  # x = image
  x = image.clone()

  for itr in range(num_iter):
      
    x.requires_grad = True

    y = net(x)

    loss = F.cross_entropy(y, label)

    net.zero_grad()

    loss.backward()

    data_grad = x.grad.data#image.grad.data
    
    # print('data grad',data_grad.norm(p=2))

    g = mu*g + (data_grad)/data_grad.norm(p=1, dim=(1,2,3), keepdim=True) 
    
    # print('g sign',g.sign().norm(p=2))

    x = x + alpha*(g.sign())
    
    x = x.detach()
    
    # print(x.norm(p=2))

  return x


def test_MIFGSM_new(model, model_attack, test_dl, epsilon, device):
    correct = 0
    
    model.eval()
    model_attack.eval()
    pbar = tqdm(test_dl,total=len(test_dl))
    
    for step, (images, labels) in enumerate(pbar, 1):
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            pred = model(images)
            
        pred_labels = pred.argmax(1)
        
        idx = labels == pred_labels
        
        images = images[idx]
        labels = labels[idx]
        
        p_images = get_MIFGSM_new(images, labels, model_attack, epsilon)
        
        # return 0
        
        with torch.no_grad():
            p_pred = model(p_images)
            
        p_pred_labels = p_pred.argmax(1)
        
        correct += (p_pred_labels == labels).sum().float().item()
        
        pbar.set_postfix({'acc': correct / (test_dl.batch_size*step)})
    
    final_acc = 100*(correct/10000.0)
    # print(correct)
    print("Epsilon: {:.2f}, Test Accuracy = {:.2f}".format(epsilon, final_acc))
    return final_acc

def test_MIFGSM(net, testloader, epsilon, num_iter, device):

    correct = 0

    net.eval()

    for image, label in testloader: 

        image, label = image.to(device), label.to(device)

        image.requires_grad = True

        output = net(image) 
       
        pred = torch.argmax(output) 

        if pred.item() != label.item(): 
          continue

        perturbed_image = get_MIFGSM(image,label,net,epsilon,num_iter) 


        output = net(perturbed_image)

        final_pred = torch.argmax(output) 

        if final_pred.item() == label.item():
            correct += 1

    final_acc = correct/float(len(testloader))
    print("Epsilon: {}, Test Accuracy = {}".format(epsilon, final_acc))

    return final_acc