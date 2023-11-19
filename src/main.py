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
import argparse
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 

    parser.add_argument('--dataset_name', default='Caltech101', type=str, help='cifar10, cifar100, gtsrb')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--numClasses', type=int, default=101, metavar='N',
                        help='number of classes(default: 101 in Caltech101)')

    parser.add_argument('--modelName',type=str, default='vit_b_16', help = 'vit_b_16,ResNet18,Vgg-16')
    parser.add_argument('--pretrained', type=str, default=True, help = 'True/ False')
    args = parser.parse_args()


    pretrained = args.pretarined
    epochs=args.epochs

    if pretrained ==True:
        layerID = 0
    else:
        layerID =7

    os.chdir(src_dir)
    import dataload
    train_loader, test_loader  = dataload.load_dataset( args.dataset_name, 6902, max_num = None)
    import train
    model = train.load_model(pretrained,layerID,args.numClasses, args.modelName)
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