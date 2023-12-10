# TransVision Analysis:
## Getting Started 
Configure the new environment:

```sh
pip install -r ./requirement.txt
```

## Dataset 
Used Caltech 101 Multi-Classification Object Dataset. The Caltech101 dataset contains images from 101 object categories (e.g., “helicopter”, “elephant” and “chair” etc.) and a background category that contains the images not from the 101 object categories. For each object category, there are about 40 to 800 images, while most classes have about 50 images. The resolution of the image is roughly about 300×200 pixels.

Compared the performace of various pretrained models like VGG-16, ResNet-18, ViT (Base 32 and Base 16) fine tuned on Caltech 101 Dataset using network based Trasnfer learning, by freezing initial layers of pretrained model and fine tuning remaining layers for target dataset. 

Compared performance of Vision Transformer like ViT-b-16 and ViT-b-32 which are trained from scratch with zero shot performance of pretrained CLIP(Vit B/32 and ViT B/16) models. CLIP stands for Constastive Language-Image Pretraining: CLIP is an open-source, multi-modal, zero-shot model. Given an image and text descriptions, the model can predict the most relevant text description for that image, without optimizing for a particular task.

## Setting up the directory
Set src and root dir in env.yml
Set all arguments in main.py as needed
## Data preprocessing
80% of our data goes for training, 9%  of which is used for validation, and reamining 20% of whole dataset used for testing. dataload.py to load the dataset and apply the transformations or data augmentation to the dataset to prevent over fitting. 

## Dataset Acquisition: 
Loading and pre-processing the data from pytorch, splitting it into train, validation and test sets. Applying transformations to the data to augment the dataset. Data augmentation techniques involve artificially modifying existing data samples to create new ones, effectively increasing the size and diversity of the training dataset. The goal is to improve the generalization ability of the machine learning model by exposing it to a wider range of data variations.

## Training and Testing
You can run cells one after the other in EE_562_project.ipynb or run the main.py file. The
train.py has the main training and test functions, path name for checkpoint and model name can be set in the train.py file.

## root_dir has model folder where the checkpoints are saved

## References
CLIP Zero shot:- https://github.com/openai/CLIP   
Caltech 101:- https://www.kaggle.com/datasets/huangruichu/caltech101   
background data preprocessing:- https://www.kaggle.com/code/dipuk0506/background-image-data   
Transfer learning with pretarined models :- https://github.com/dipuk0506/SpinalNet/blob/master/Transfer%20Learning/Transfer_Learning_STL10.py
Tranining scripts :- https://www.kaggle.com/code/dipuk0506/caltech101-transformer-background/notebook
