from torchvision import datasets,transforms
import numpy as np
import math
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import yaml
import argparse
import shutil
import torch.nn as nn
import torch.nn.functional as F
import clip

config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']

classes = os.listdir(root_dir + '/data/test/')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
transform = preprocess

data = {
    'train': datasets.ImageFolder(root=root_dir +'/data/train'),
    'eval': datasets.ImageFolder(root=root_dir+'/data/eval'),
    'test': datasets.ImageFolder(root=root_dir+'/data/test')
}
test_set = data['test']
cnt = 0
total_cnt = len(test_set)
for image_input, class_id in test_set:
  # print(class_id)
  # print(classes[class_id])  # # from PIL import Image

  # # # # Preprocess the image
  image_input = preprocess(image_input).unsqueeze(0).to(device)
  text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

  # # # Calculate features
  with torch.no_grad():
      image_features = model.encode_image(image_input)
      text_features = model.encode_text(text_inputs)

  # Pick the top 5 most similar labels for the image
  image_features /= image_features.norm(dim=-1, keepdim=True)
  text_features /= text_features.norm(dim=-1, keepdim=True)
  similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
  print(similarity)
  values, indices = similarity[0].topk(1)

  # Print the result
  # print("\nTop predictions:\n")
  for value, index in zip(values, indices):
      # print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
      if(index == class_id):
        cnt= cnt+1
        print('image_input',image_input)
        print('class_id',class_id)
print("Accuracy of clip_B_32 : ",cnt/total_cnt)
