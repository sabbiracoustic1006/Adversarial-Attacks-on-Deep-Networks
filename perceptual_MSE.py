# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 20:18:07 2022

@author: user
"""

import torch 
import torch.nn as nn
from torchvision.models.vgg import vgg16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class perceptual_loss(nn.Module):
    def __init__(self):
        super(perceptual_loss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:21]).eval()

        for param in loss_network.parameters():
            param.requires_grad = True
        self.loss_network = loss_network.to(device)        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, imgs1, imgs2):        
        if imgs1.shape[1] == 1:
            out = torch.cat((imgs1,imgs1,imgs1),dim=1)
            tar = torch.cat((imgs2,imgs2,imgs2),dim=1)
        else:
            out = imgs1
            tar = imgs2
        # print(self.loss_network(out).shape)
        
        Loss = self.mse_loss(self.loss_network(out), self.loss_network(tar))
        return Loss 