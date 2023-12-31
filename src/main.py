import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
import yaml
import argparse

# defining src and root dir
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']
    
import dataload
from dataload import *
import train
from train import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 

    parser.add_argument('--dataset_name', default='Caltech101', type=str, help='cifar10, cifar100, gtsrb')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--numClasses', type=int, default=102, metavar='N',
                        help='number of classes(default: 102 in Caltech101 including background)')

    parser.add_argument('--modelName',type=str, default='vit_b_32', help = 'vit_b_16,vit_b_32,ResNet18,Vgg-16')
    parser.add_argument('--pretrained', type=str, default='False', help = 'True/ False')
    args = parser.parse_args(args=['--dataset_name', 'Caltech101', '--epochs', '100', '--numClasses', '102', '--modelName', 'vit_b_32', '--pretrained', 'False'])



    pretrained = args.pretrained
    epochs=args.epochs

    if pretrained == 'True':
        print('from pretrained')
        pretrained = True
        layerID = 7
    else:
        pretrained = False
        print('from scratch')
        layerID = 0

    os.chdir(src_dir)
    import dataload
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loading dataset
    train_loader, test_loader, dataset_sizes, class_names   = dataload.load_dataset( args.dataset_name, 9000, max_num = None)
    #loading model architecture
    model_ft = train.load_model(pretrained,layerID,args.numClasses, args.modelName)
    model_ft = model_ft.to(device)

    #define optimizer used during training process
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #training the model
    model_ft = train.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader,dataset_sizes, epochs) 

    cluster = dataset_sizes['train']
    delta = {}
    
    # save path
    save_path = root_dir +'/model/'+ args.modelName + '_trSize_' + str(dataset_sizes['train'])  + '_Epoch_'+ str(args.epochs) +  '_Test.pt')

    # testing the model
    acc_test= train.test_model(model_ft, test_loader, args.numClasses, 'test')
                        
    delta['acc_test'] = acc_test
    # saving results and final model                       
    np.save(root_dir + '/results_0.0005'+'_'+'_'+str(layerID)+'_'+str(cluster)+'_'+str(epochs)+'_.npy', delta)
    torch.save(model_ft.state_dict(), save_path)
