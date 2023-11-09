import torch
import torch.optim as optim
from torchvision.models import swin_t
import dataload
import os
import yaml
from utils import mkdir_p
config_file = '../env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
    src_dir = yamlfile['src_dir']


# Define the Swin-Transformer model
model = swin_t(num_classes=102)


#Test Function
def test(model, test_loader, verbos=True):

	test_loss = 0
	correct = 0
	model.eval()
	with torch.no_grad():
		for data, target in test_loader:
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
  for batch_idx, (data, target) in enumerate(train_loader):
      # print(batch_idx, " - ",f"Data shape: {data.shape}",f"Target shape: {target.shape}")
      optimizer.zero_grad()
      output = model(data)
      loss = F.cross_entropy(output, target)
      loss.backward()
      optimizer.step()

  acc= test(model,train_loader,verbos=False)
  print('TargetModel Train Epoch: {} \tLoss: {:.6f}, \t Accuracy: {:.2f}'.format(epoch,loss.item(),acc))

  return acc

##training model and saving checkpoint 
def Train_Model(model, train_loader,test_loader,pretrained, epochs):
	cluster = len(train_loader.dataset)
	description= ''
	if pretrained ==False:
		description+='swin_t_NoTransferLearning'
	else:
		description+='_LayerID_'+str(layerID)
	save_path= root_dir +'/model/'+	description

	if not os.path.exists(save_path+'_Epoch_'+str(epochs)):
		print('***'*10,'Saving model to: ', save_path+'_Epoch_'+str(epochs))
		model.train()
		optimizer = optim.Adam(model.parameters(), lr=0.01)
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
		for epoch in range(1, epochs + 1):
			train( model, train_loader, optimizer, epoch)
			if epoch%100==0:
				torch.save(model.state_dict(), save_path+'_Epoch_'+str(epoch))
		torch.save(model.state_dict(), save_path+'_Epoch_'+str(epochs))
	else:
		print(save_path+'_Epoch_'+str(epochs))
		print('***'*10,'Loading model from: ', save_path+'_Epoch_'+str(epochs))
		model.load_state_dict(torch.load(save_path+'_Epoch_'+str(epochs)))

	# test(model, test_loader)

	return model