from torchvision import datasets,transforms
import numpy as np
import math
import torch
import os
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import yaml
import argparse
import shutil

import clip

config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']
  
classes = os.listdir(root_dir + '/data/train/')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)
transform = preprocess

data = {
    'train': datasets.ImageFolder(root=root_dir +'train'),
    'eval': datasets.ImageFolder(root=root_dir+'eval'),
    'test': datasets.ImageFolder(root=root_dir+'test')
}
test_set = data['test']

correct_predictions = 0
total_predictions = 0


for i in range(len(test_set)):

  image_input, class_id = test_set[10]

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
  values, indices = similarity[0].top(5)
    # Get the predicted class
  predicted_class = similarity[0].argmax().item()

    # Check if the prediction is correct
  if predicted_class == class_id:
      correct_predictions += 1

  total_predictions += 1

# Calculate accuracy
accuracy = (100.0 * correct_predictions) / total_predictions
print(f"\nAccuracy: {accuracy:.2f}%")
