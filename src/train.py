import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.nn.functional as F

import dataload
import os
import yaml
from utils import mkdir_p
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']


# Define the ViT-B-16 model
def load_model(pretrained,layerid,NUM_CLASSES, model_name):
	if(model_name == 'vit_b_16'):
		model = vit_b_16(pretrained = False)
		model=layerFreezing(model,layerid,NUM_CLASSES, model_name, pretrained)
		return model

def layerFreezing(model,layerid,NUM_CLASSES, model_name='vit_b_16', pretrained=True):

	
	"""
	While implementing Transfer Learning, a certain number of layers in the pretrained model are frozen.
	This function is called to choose the number of frozen layers from 0 to N-2. TL0 indictaes that the 
	weights of the Pretrained model are used to train the Target model.
	"""
	
	if pretrained == False: 
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
							  nn.Linear(512, 102),)		
	return model

#Test Function
def test(model, test_loader, verbos=True):

	test_loss = 0
	correct = 0
	model.eval()
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.cuda(), target.cuda()
			output = model(data)
			test_loss += F.cross_entropy(output, target).item()
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	accuracy = 100. * correct / len(test_loader.dataset)
	if verbos:
		print('\n Test Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset), accuracy))
	return accuracy

##Train function
def train(model, train_loader, optimizer, epoch):
  cnt = 0
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.cuda(), target.cuda()
      # print(batch_idx, " - ",f"Data shape: {data.shape}",f"Target shape: {target.shape}")
      optimizer.zero_grad()
      output = model(data)
      loss = F.cross_entropy(output, target)
      loss.backward()
      optimizer.step()
      cnt +=1
  print('size of training data covered', cnt)
  acc= test(model,train_loader,verbos=False)
  print('TargetModel Train Epoch: {} \tLoss: {:.6f}, \t Accuracy: {:.2f}'.format(epoch,loss.item(),acc))

  return acc

##training model and saving checkpoint 
def Train_Model(model, train_loader,test_loader,pretrained, epochs):
	layerID = 10
	torch.cuda.empty_cache()
	cluster = len(train_loader.dataset)
	description= ''
	if pretrained ==False:
		description+='vitb16_trSize_' + str(cluster)
	else:
		description+='vitb16_LayerID_'+str(layerID)
	save_path= root_dir +'/model/'+	description

	if not os.path.exists(save_path+'_Epoch_'+str(epochs)):
		print('***'*10,'Saving model to: ', save_path+'_Epoch_'+str(epochs))
		model.train()
		model.cuda()
		optimizer = optim.Adam(model.parameters(), lr=0.01)
		# optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
		for epoch in range(1, epochs + 1):
			train( model, train_loader, optimizer, epoch)
			if epoch%100==0:
				torch.save(model.state_dict(), save_path+'_Epoch_'+str(epoch))
		torch.save(model.state_dict(), save_path+'_Epoch_'+str(epochs))
	else:
		print(save_path+'_Epoch_'+str(epochs))
		print('***'*10,'Loading model from: ', save_path+'_Epoch_'+str(epochs))
		model.load_state_dict(torch.load(save_path+'_Epoch_'+str(epochs)))
		model.cuda()		
	test(model, test_loader)

	return model