from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import math
import yaml
import zipfile

import torch
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _accumulate
from torch.utils.data import TensorDataset, Subset, DataLoader, ConcatDataset

from zipfile import ZipFile

config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

#function to load dataset
def load_dataset( dataset, cluster=6902, max_num = None):
    config_file = '../env.yml'
    with open(config_file, 'r') as stream:
        yamlfile = yaml.safe_load(stream)
        root_dir = yamlfile['root_dir']
        src_dir = yamlfile['src_dir']
    kwargs = {'num_workers': 2, 'pin_memory': True}
    #data augmentation for simple datasets like cifar10, gtsrb
    transform = transforms.Compose([
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
        whole_set = datasets.Caltech101(root=root_dir, download = True)
        
        
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
        #Preprocessing of dataset by applying data augmentation technique      
        im_dimention = 224

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((244,244)),
                transforms.RandomRotation(15,),
                transforms.RandomCrop(im_dimention),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'eval': transforms.Compose([
                transforms.Resize((im_dimention,im_dimention)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'test': transforms.Compose([
                transforms.Resize((im_dimention,im_dimention)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
        }

        newpath =root_dir + '/data/'
        oldPath = root_dir +'/caltech101/101_ObjectCategories'
        classes = os.listdir(oldPath)
        
        # spliting data into train, val, test set
        if not os.path.isdir(newpath +'train'):
            train_test_validation_Split(oldPath, newpath, classes)
        elif not os.listdir(newpath +'train'):
            train_test_validation_Split(oldPath, newpath, classes)

        count = 0
        for root_dir, cur_dir, files in os.walk(newpath+'train/'):
            count += len(files)
        print('Train image count without Background Class:', count)

        # data augmentation by adding background data to train and eval set but not to test set
        for iterf in ['train/', 'eval/']:
            create_dir(newpath + iterf + 'zzzBackground') 

        create_dir(newpath +'Background')
        with zipfile.ZipFile(newpath+'Background_data.zip', 'r') as zip_ref:
            zip_ref.extractall(newpath + 'Background/')

        image_index = 0
        for dirpath, dirnames, filenames in os.walk(newpath +'Background/'):
            for fname in filenames:
                if image_index == 2000: 
                    shutil.copyfile(path, newpath + 'eval/'+'zzzBackground'+'/'+str(image_index) + fname[-9:])
                    #break 
                path = os.path.join(dirpath,fname)
                shutil.copyfile(path, newpath + 'train/'+'zzzBackground'+'/'+str(image_index) + fname[-9:])
                image_index = image_index +1
        count = 0

        for root_dir, cur_dir, files in os.walk(newpath+'train/'):
            count += len(files)
        print('Image count with Background Class:', count)      

        

    #using dataloader 
    if(dataset == 'Caltech101'):
        data = {x: datasets.ImageFolder(os.path.join(newpath, x),
                                        data_transforms[x])
                for x in ['train', 'eval','test']}
        dataset_sizes = {x: len(data[x]) for x in ['train', 'eval']}
        class_names = data['train'].classes

        train_loader = {x: torch.utils.data.DataLoader(data[x], batch_size=16,
                                            shuffle=True, num_workers=0)
            for x in ['train', 'eval']}

        test_loader = {'test': torch.utils.data.DataLoader(data['test'], batch_size=16,
                                            shuffle=False, num_workers=0)}
    else:
        train_loader = DataLoader(train_set, batch_size= 16, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size= 16, shuffle=False, **kwargs)
    return train_loader, test_loader, dataset_sizes, class_names 

def create_dir(folderlocation):
    if not os.path.exists(folderlocation):
        os.mkdir(folderlocation)

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

# function to split whole dataset into train, test or eval data 
def train_test_validation_Split(oldpath, newpath, classes):
    for name in classes:
        if(name != 'BACKGROUND_Google'):
            print('class_name',name)
            full_dir = os.path.join(os.getcwd(), f"{oldpath}/{name}")

            files = list_files(full_dir)
            total_file = np.size(files,0)
            # We split data set into 3: train, validation and test
            
            train_size = math.ceil(total_file * 71/100) # 71% for training 

            validation_size = train_size + math.ceil(total_file * 9/100) # 9% for validation
            test_size = validation_size + math.ceil(total_file * 20/100) # 20% for testing 
            
            train = files[0:train_size]
            validation = files[train_size:validation_size]
            test = files[validation_size:]
            movefiles(train, full_dir,newpath + f"train/{name}")
            movefiles(validation, full_dir,newpath+ f"eval/{name}")
            movefiles(test, full_dir,newpath+ f"test/{name}")
            
# function to move files from download path to correposnding train, test or eval data path
def movefiles(files, old_dir, new_dir):    

    new_dir = os.path.join(os.getcwd(), new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = os.path.join(os.getcwd(), f"{old_dir}/{file}")
        new_file_path = os.path.join(os.getcwd(), f"{new_dir}/{file}")

        shutil.copyfile(old_file_path, new_file_path)
