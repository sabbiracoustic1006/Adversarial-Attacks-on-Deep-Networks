# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:08:16 2022

@author: user
"""

import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score


def SEED_EVERYTHING(seed_val=0):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  
  
class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count
      
      
@torch.no_grad()
def evaluate(model, test_dl, device, loss_func):
  pred_labels = []
  tgt_labels = []

  model.eval()
  loss_meter = AverageMeter()
  for step, batch in enumerate(test_dl):
    img = batch[0].to(device)
    tgt = batch[1].to(device)
    
    pred = model(img)

    loss = loss_func(pred, tgt)
    loss_meter.update(loss.item())
    
    pred_label = pred.argmax(1).cpu().numpy()
    tgt_label = batch[1].numpy()

    pred_labels.append(pred_label)
    tgt_labels.append(tgt_label)
    
  pred_labels = np.concatenate(pred_labels)
  tgt_labels = np.concatenate(tgt_labels)
  acc = accuracy_score(pred_labels, tgt_labels)
  return loss, acc