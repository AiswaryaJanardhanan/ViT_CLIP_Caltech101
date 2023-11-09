from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import math
import yaml
from utils import mkdir_p

import torch
from torch import randperm
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _accumulate
from torch.utils.data import TensorDataset, Subset, DataLoader, ConcatDataset
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

def load_dataset( dataset, cluster=1000, max_num = None):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((224, 224)), # Resize to 224x224 (height x width)
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                       std=[0.229, 0.224, 0.225])
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    ])
                           
    if dataset == 'cifar10':
        print("Loading CIFAR10 Dataset")
        whole_set = datasets.CIFAR10(root_dir, train=True, download=True, transform=transform)
        

    elif dataset == 'cifar100':
        print("Loading CIFAR100 Dataset")
        whole_set = datasets.CIFAR100(root_dir, train=True, download=True, transform=transform)
        

    elif dataset == 'gtsrb':
        print("Loading GTSRB Dataset")
        whole_set = datasets.GTSRB(root=root_dir, split= 'train', transform=transform, download = True)
        
    elif dataset == 'Caltech101':
        print("Loading Caltech101 Dataset")
        whole_set = datasets.Caltech101(root=root_dir, transform=transform, download = True)
        
        
    if dataset != 'Caltech101':
      length= len(whole_set)
      train_size = round(0.8*length)
      test_size = round(0.2*length)
      # test_size = int(round(0.2*length))
      remain_size = length - train_size - test_size
      train_set, _, test_set = dataset_split(whole_set, [train_size, remain_size, test_size])
      if not max_num is None:
          print(max_num)
          train_set = train_set[:max_num]
          test_set = test_set[:max_num]
    else:
      
      image_transforms = {
            'train': transforms.Compose([
                      transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                      transforms.RandomRotation(degrees=15),
                      transforms.RandomHorizontalFlip(),
                      transforms.CenterCrop(size=224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]),
            'validation': transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])              
            ]),
            'test': transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])              
            ])
      }

      newpath =root_dir + '/caltech101/'
      oldPath = root_dir +'/caltech101/101_ObjectCategories'
      classes = os.listdir(oldPath)

      train_test_validation_Split(oldPath, newpath, classes)

      data = {
          'train': datasets.ImageFolder(root=root_dir +'/caltech101/train', transform=image_transforms['train']),
          'validation': datasets.ImageFolder(root=root_dir+'/caltech101/validation',transform=image_transforms['validation']),
          'test': datasets.ImageFolder(root=root_dir+'/caltech101/test', transform=image_transforms['test'])
      }
      
      transform=image_transforms['train']
      train_set = data['train']
      validation_data= data['validation']
      test_set = data['test']
      
    train_loader = DataLoader(train_set, batch_size= 32, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size= 32, shuffle=False, **kwargs)
    return train_loader, test_loader    


def dataset_split(dataset, lengths,seed=1):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    print(sum(lengths))

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = list(range(sum(lengths)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

# Get all files in the dataset directory
def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)

def train_test_validation_Split(oldpath, newpath, classes):
    for name in classes:
        full_dir = os.path.join(os.getcwd(), f"{oldpath}/{name}")

        files = list_files(full_dir)
        total_file = np.size(files,0)
        # We split data set into 3: train, validation and test
        
        train_size = math.ceil(total_file * 3/4) # 75% for training 

        validation_size = train_size + math.ceil(total_file * 1/8) # 12.5% for validation
        test_size = validation_size + math.ceil(total_file * 1/8) # 12.5x% for testing 
        
        train = files[0:train_size]
        validation = files[train_size:validation_size]
        test = files[validation_size:]

        movefiles(train, full_dir,newpath + f"train/{name}")
        movefiles(validation, full_dir,newpath+ f"validation/{name}")
        movefiles(test, full_dir,newpath+ f"test/{name}")

def movefiles(files, old_dir, new_dir):
    new_dir = os.path.join(os.getcwd(), new_dir);
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = os.path.join(os.getcwd(), f"{old_dir}/{file}")
        new_file_path = os.path.join(os.getcwd(), f"{new_dir}/{file}")

        shutil.move(old_file_path, new_file_path)


        
