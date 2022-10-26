# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:08:01 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self, dataset='mnist'):
        super(Net, self).__init__()
        
        if dataset == 'mnist':
            inp_channels = 1
            num_classes = 10
            num_features = 64
        else:
            inp_channels = 3
            num_classes = 100
            num_features = 256
            
        self.conv1 = nn.Sequential(nn.Conv2d(inp_channels, 16, kernel_size=3, stride = 1, padding =1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride = 1, padding =1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding =1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding =1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(2,2)

        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    

class ConvNet(nn.Module):
    def __init__(self, init_weights=False):
        super(ConvNet, self).__init__()
        #INITIALIZE LAYERS HERE

        # define layers
        self.conv1 = nn.Sequential(nn.Conv2d(3,32,3,1,1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(2,2))                                   
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,1,1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(2,2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(2,2))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,1,1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(2,2))
        self.conv5 = nn.Sequential(nn.Conv2d(256,512,3,1,1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(2,2))
        
        self.drop = nn.Dropout(0.25)
        
        self.linear1 = nn.Linear(512,64)
        
        self.linear2 = nn.Linear(64,10)
        
        self._init_weights()
        
    
        
    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
    

    # define forward function
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(x.size(0),-1)
        
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x) 
      
  
        return x
          

class ResNet20(nn.Module):
    def __init__(self, dataset='mnist'):
        super(ResNet20, self).__init__()
        
        if dataset == 'mnist':
            inp_channels = 1
            num_classes = 10
        else:
            inp_channels = 3
            num_classes = 100
        
        self.conv1 = nn.Sequential(nn.Conv2d(inp_channels,16,3,1,1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace = True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(16,16,3,1,1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(16,16,3,1,1),
                                   nn.BatchNorm2d(16))
        
        self.conv3 = nn.Sequential(nn.Conv2d(16,32,3,2,1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32,32,3,1,1),
                                   nn.BatchNorm2d(32))
        
        self.conv4 = nn.Sequential(nn.Conv2d(16,32,1,2,0),
                                   nn.BatchNorm2d(32))
        
        self.conv5 = nn.Sequential(nn.Conv2d(32,32,3,1,1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32,32,3,1,1),
                                   nn.BatchNorm2d(32))
        
        self.conv6 = nn.Sequential(nn.Conv2d(32,64,3,2,1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64,64,3,1,1),
                                nn.BatchNorm2d(64))
        
        self.conv7 = nn.Sequential(nn.Conv2d(32,64,1,2,0),
                                nn.BatchNorm2d(64))
        
        self.conv8 = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64,64,3,1,1),
                                nn.BatchNorm2d(64))
        
        self.linear = nn.Linear(64,num_classes)
        
        
      
        
        
        self._init_weights()
          
    
        
    def _init_weights(self):
          
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
      


    def forward(self, x):
    
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x1 + x2
        x2 = F.relu(x2)
        x3 = F.relu(self.conv2(x2) + x2)
      
        x4 = F.relu(self.conv2(x3) + x3)
      
        x5 = F.relu(self.conv3(x4) + self.conv4(x4))
      
        x6 = F.relu(self.conv5(x5) + x5)
      
        x7 = F.relu(x6 + self.conv5(x6))
      
        x8 = F.relu(self.conv6(x7) + self.conv7(x7))
      
        x9 = F.relu(self.conv8(x8) + x8)
      
        x10 = F.relu(self.conv8(x9) + x9)

        x11 = F.avg_pool2d(x10, x10.size()[3])
        
        x11 = x11.view(x11.size(0), -1)
        
        out = self.linear(x11)
      
        return out

class Vgg16(nn.Module):
    def __init__(self, dataset='mnist'):
        super(Vgg16, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
            
        
        if dataset == 'mnist':
            vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            vgg16.features = vgg16.features[:30]
            num_classes = 10
        else:
            num_classes = 100
        
        self.feature_extractor = vgg16.features
        self.classifier = nn.Linear(512, num_classes)
                
                
    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x.view(x.shape[0],-1))
        return out
    
    
class DenseNet(nn.Module):
    def __init__(self, dataset='mnist'):
        super(DenseNet, self).__init__()
        
        densenet121 = models.densenet121(pretrained=True)
            
        
        if dataset == 'mnist':
            densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                                   stride=(1, 1), padding=(3, 3),
                                                   bias=False)
            num_classes = 10
            densenet121.classifier = nn.Linear(1024, num_classes)
        else:
            num_classes = 100
            densenet121.classifier = nn.Linear(1024, num_classes)
            
        self.densenet121 = densenet121
        
                
                
    def forward(self, x):
        out = self.densenet121(x)
        return out
    
def get_model(model_name, dataset):
    if model_name == 'vanilla':
        if dataset == 'mnist':
            model = Net(dataset)
        else:
            model = ConvNet()
    elif model_name == 'resnet20':
        model = ResNet20(dataset)
    elif model_name == 'densenet121':
        model = DenseNet(dataset)
    elif model_name == 'vgg16':
        model = Vgg16(dataset)
    return model        
                

if __name__ == '__main__':
    mnist_model = DenseNet(dataset='mnist')
    cifar_model = DenseNet(dataset='cifar')
    
    mnist_model.eval()
    cifar_model.eval()
    
    inp_mnist = torch.rand(1,1,28,28)
    inp_cifar = torch.rand(1,3,32,32)
    
    out_mnist = mnist_model(inp_mnist)
    print(out_mnist.shape)
    
    out_cifar = cifar_model(inp_cifar)
    print(out_cifar.shape)


