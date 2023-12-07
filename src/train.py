import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torchvision.models import vit_b_16,vit_b_32
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dataload
import os
import yaml, time, copy
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']


# Define the loading model function ViT-B-16 model
def load_model(pretrained,layerid,NUM_CLASSES, model_name):
	if(model_name == 'vit_b_16'):

		if pretrained == True:
			print('from pretrained')
			model = vit_b_16(weights='DEFAULT')
		else:
			print('from scratch')
			model = vit_b_16(pretrained =False)
		
		model=layerFreezing(model,layerid,NUM_CLASSES, model_name, pretrained)
		
		return model
	
	if(model_name == 'vit_b_32'):
		if pretrained == True:
			print('from pretrained')
			model = vit_b_32(weights='DEFAULT')
		else:
			print('from scratch')
			model = vit_b_32(pretrained =False,num_classes= NUM_CLASSES)
		return model

# layer freezing function for transfer learning replacing last layer model head to have same output as number of classes 
def layerFreezing(model,layerid,NUM_CLASSES, model_name='vit_b_16', pretrained=False):

	
	"""
	While implementing Transfer Learning, a certain number of layers in the pretrained model are frozen.
	This function is called to choose the number of frozen layers from 0 to N-2. TL0 indictaes that the 
	weights of the Pretrained model are used to train the Target model.
	"""
	
	if pretrained == 'False': 
		layerid=0
		
	count=0
	for param in model.parameters():
		count+=1
		param.requires_grad = False
	print(count)
	
	if layerid > count:
		print("The layerid should be in range of 0 to "+ str(count))
		quit()
		return -1
	for  i,param in enumerate(model.parameters()):
		if i>=layerid:
			param.requires_grad = True

	if 'vgg' in model_name:
		model.classifier[3].requires_grad=True
		model.classifier[6]= nn.Sequential(
							  nn.Linear(4096, 512), 
							  nn.ReLU(), 
							  nn.Dropout(0.5),
							  nn.Linear(512, NUM_CLASSES))
	elif 'swin_t' in model_name:
		model.head = nn.Sequential(
							  nn.Linear(768, 512),
							  nn.ReLU(),
							  nn.Dropout(0.5),
							  nn.Linear(512, NUM_CLASSES),)
	elif model_name == 'resnet18':
		model.fc= nn.Linear(512, NUM_CLASSES)
		
	elif model_name == 'resnet50':
		model.fc= nn.Linear(2048, NUM_CLASSES)
		
	elif model_name == 'facenet':
		model.fc= nn.Linear(512, NUM_CLASSES)
  
	elif model_name == 'vit_b_16':
		model.head = nn.Sequential(
							  nn.Linear(768, 512),
							  nn.ReLU(),
							  nn.Dropout(0.5),
							  nn.Linear(512, NUM_CLASSES),)		
	elif model_name == 'vit_b_32':
		model.head = nn.Sequential(
							  nn.Linear(768, 512),
							  nn.ReLU(),
							  nn.Dropout(0.5),
							  nn.Linear(512, NUM_CLASSES),)		
									
	return model

# traing model function
def train_model(model, criterion, optimizer, scheduler,dataloaders,dataset_sizes, num_epochs=25):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start = time.time()

    bestModel = copy.deepcopy(model.state_dict())
    bestModelAcc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation mode
        for mode in ['train', 'eval']:
          
            if mode == 'train':
				# Set model to training mode
                model.train()  
            else:
				# Set model to evaluate mode
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training mode
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if mode == 'train':
                scheduler.step()

            crntEpochLoss = running_loss / dataset_sizes[mode]
            crntEpochAcc = running_corrects.double() / dataset_sizes[mode]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                mode, crntEpochLoss, crntEpochAcc))

            # deep copy the model
            if mode == 'eval' and crntEpochAcc > bestModelAcc:
                bestModelAcc = crntEpochAcc
                bestModel = copy.deepcopy(model.state_dict())
        print()
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(bestModelAcc))

    # load best model weights
    model.load_state_dict(bestModel)

	#saving model 
    save_path = root_dir +'/model/'+ 'vitb16_trSize_' + str(dataset_sizes['train'])
    torch.save(model.load_state_dict(bestModel), save_path+'_Epoch_'+str(num_epochs))

    return model

# function to test model
def test_model(model_ft, dataloaders_test, Num_class, mode = 'test'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	y_pred = []
	y_true = []
	output_all =[]

	# iterate over test data
	for inputs, labels in dataloaders_test[mode]:
			inputs = inputs.to(device)
			labels = labels.to(device)
			
			output = model_ft(inputs) # Feed Network
			output = output[:,0:Num_class] # Discarding Background Class
			output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
			y_pred.extend(output) # Save Prediction
			
			labels = labels.data.cpu().numpy()
			y_true.extend(labels) # Save Truth
			
			
	test_accuracy = 0
	for iter1 in range(len(y_true)):
		if y_true[iter1] == y_pred[iter1]:
			test_accuracy = test_accuracy + 1
	acc = test_accuracy/len(y_true)
	print(mode,'Accuracy:',acc)

	return acc
