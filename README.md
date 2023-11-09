# TransVision Analysis:
## Getting Started 
Configure the new environment:

```sh
pip install -r ./requirement.txt
```

## Dataset 
We will use Caltech 101 Multi-Classification Object Dataset. The Caltech101 dataset contains images from 101 object categories (e.g., “helicopter”, “elephant” and “chair” etc.) and a background category that contains the images not from the 101 object categories. For each object category, there are about 40 to 800 images, while most classes have about 50 images. The resolution of the image is roughly about 300×200 pixels.

We will compare the performace of various models like VGG-16, ResNet-18, ViT (swin transformer) on  1000 samples of CIFAR10 and GTSRB dataset to analyse the imapce of Transfer learning on the performance of the model. CIFAR10 dataset contains 60000 images of 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. GTSRB dataset contains more than 50,000 images of 43 classes of traffic signs. The images are in different shapes because of the different shapes of the traffic signs. The images are in RGB format. The size of the images is 32x32 pixels. 

## Setting up the directory
Set src and root dir in env.yml
Set src_dir in EE_562_project.ipynb

## Data preprocessing
75% of our data goes for training, 12.5% for validation, and 12.5% for testing. We will use the dataload.py to load the dataset and apply the transformations to the dataset.

## Dataset Acquisition: 
Loading and pre-processing the data from pytorch, splitting it into train, validation and test sets. Applying transformations to the data to augment the dataset. Data augmentation techniques involve artificially modifying existing data samples to create new ones, effectively increasing the size and diversity of the training dataset. The goal is to improve the generalization ability of the machine learning model by exposing it to a wider range of data variations.

## Training and Testing
You can run cells one after the other in EE_562_project.ipynb
train.py has the main training and test functions, path name for checkpoint and model name can be set in the train.py file.

## root_dir has model folder where the checkpoints are saved
