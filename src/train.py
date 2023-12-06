import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torchvision.models import vit_b_16
from torchvision.models import vit_b_32
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
# root_dir = '/content/drive/MyDrive/ECE_562_Project/root'
# src_dir = '/content/drive/MyDrive/ECE_562_Project/src'


# Define the ViT-B-16 model
def load_model(pretrained, layerid, NUM_CLASSES, model_name):
    if model_name == 'vit_b_16':
        if pretrained:
            model = vit_b_16(weights='DEFAULT')
        else:
            model = vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        # Assuming a similar function for ViT-B-32 as ViT-B-16
        if pretrained:
            model = vit_b_32(weights='DEFAULT')  # Replace 'vit_b_32' with your actual function
        else:
            model = vit_b_32(weights=None)      # Replace 'vit_b_32' with your actual function
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model = layerFreezing(model, layerid, NUM_CLASSES, model_name, pretrained)
    return model

def layerFreezing(model,layerid,NUM_CLASSES, model_name='vit_b_16', pretrained=False):

	
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
  
	elif model_name in ['vit_b_16', 'vit_b_32']:
		model.head = nn.Sequential(
							  nn.Linear(768, 512),
							  nn.ReLU(),
							  nn.Dropout(0.5),
							  nn.Linear(512, NUM_CLASSES),)		

                
	return model

# #Test Function
# def test(model, test_loader, verbos=True):

# 	test_loss = 0
# 	correct = 0
# 	model.eval()
# 	with torch.no_grad():
# 		for data, target in test_loader:
# 			data, target = data.cuda(), target.cuda()
# 			output = model(data)
# 			test_loss += F.cross_entropy(output, target).item()
# 			pred = output.max(1, keepdim=True)[1]
# 			correct += pred.eq(target.view_as(pred)).sum().item()

# 	test_loss /= len(test_loader.dataset)

# 	accuracy = 100. * correct / len(test_loader.dataset)
# 	if verbos:
# 		print('\n Test Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
# 			test_loss, correct, len(test_loader.dataset), accuracy))
# 	return accuracy

# ##Train function
# def train(model, train_loader, optimizer, epoch):
#   cnt = 0
#   for batch_idx, (data, target) in enumerate(train_loader):
#       data, target = data.cuda(), target.cuda()
#       # print(batch_idx, " - ",f"Data shape: {data.shape}",f"Target shape: {target.shape}")
#       optimizer.zero_grad()
#       output = model(data)
#       loss = F.cross_entropy(output, target)
#       loss.backward()
#       optimizer.step()
#       cnt +=1
#   print('size of training data covered', cnt)
#   acc= test(model,train_loader,verbos=False)
#   print('TargetModel Train Epoch: {} \tLoss: {:.6f}, \t Accuracy: {:.2f}'.format(epoch,loss.item(),acc))

#   return acc

##training model and saving checkpoint 
# def Train_Model(model, train_loader,test_loader,pretrained, epochs):
# 	layerID = 0
# 	torch.cuda.empty_cache()
# 	cluster = len(train_loader.dataset)
# 	description= ''
# 	if pretrained ==False:
# 		description+='vitb16_trSize_' + str(cluster)
# 	else:
# 		layerID =7
# 		description+='vitb16_LayerID_'+str(layerID)
# 	save_path= root_dir +'/model/'+	description
# 	if not os.path.exists(save_path+'_Epoch_'+str(epochs)):
# 		print('***'*10,'Saving model to: ', save_path+'_Epoch_'+str(epochs))
# 		model.train()
# 		model.cuda()
# 		optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 		# optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
# 		for epoch in range(1, epochs + 1):
# 			train( model, train_loader, optimizer, epoch)
# 			if epoch%100==0:
# 				torch.save(model.state_dict(), save_path+'_Epoch_'+str(epoch))
# 		torch.save(model.state_dict(), save_path+'_Epoch_'+str(epochs))
# 	else:
# 		print(save_path+'_Epoch_'+str(epochs))
# 		print('***'*10,'Loading model from: ', save_path+'_Epoch_'+str(epochs))
# 		model.load_state_dict(torch.load(save_path+'_Epoch_'+str(epochs)))
# 		model.cuda()		
# 	test(model, test_loader)

# 	return model

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