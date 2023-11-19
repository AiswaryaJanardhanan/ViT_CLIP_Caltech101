import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
from torchvision import datasets, transforms
import yaml
from utils import mkdir_p


config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']
# root_dir = '/content/drive/MyDrive/ECE_562_Project/root'
# src_dir = '/content/drive/MyDrive/ECE_562_Project/src'
pretrained = True
epochs=500

if pretrained ==True:
    layerID = 0
else:
    layerID =7

os.chdir(src_dir)
import dataload
train_loader, test_loader  = dataload.load_dataset( 'Caltech101', 6902, max_num = None)
import train
model = train.load_model(pretrained,layerID,101, 'vit_b_16')
model = train.Train_Model(model, train_loader,test_loader,pretrained,epochs)

cluster = len(train_loader.dataset)
delta = {}
# model.load_state_dict(torch.load(root_dir + '/model/'+'_Epoch_'+str(args.epochs)))
# model.cuda()
model.eval()
acc_test= train.test(model, test_loader, verbos=True)
acc_train= train.test(model, train_loader, verbos=True)
                    
delta['acc_test'] = acc_test
delta['acc_train'] = acc_train
                        
np.save(root_dir + '/results_0.0001'+'_'+'_'+str(layerID)+'_'+str(cluster)+'_'+str(epochs)+'_.npy', delta)