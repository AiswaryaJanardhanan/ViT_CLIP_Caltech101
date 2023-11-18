import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
from torchvision import datasets, transforms
src_dir = './'
os.chdir(src_dir)
import dataload
train_loader, test_loader  = dataload.load_dataset( 'Caltech101', 1000, max_num = None)
import train 
model = train.load_model(False,10,102, 'vit_b_16')
print(model)
train.Train_Model(model, train_loader,test_loader,False,epochs=1)