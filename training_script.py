from AlexNetPytorch import*
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch
from IPython.core.debugger import set_trace
import os
import visdomvisualize


PATH = "savedmodel.tar"


AlexNet = AlexNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(AlexNet.parameters(), lr=0.001, momentum=0.9)

if os.path.exists("savedmodel.tar"):
  checkpoint = torch.load(PATH)
  AlexNet.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  print("checkpoint loaded")

all_data = np.load('training_data2.npy')
inputs= all_data[:,0]
labels= all_data[:,1]
inputs_tensors = torch.stack([torch.Tensor(i) for i in inputs])
labels_tensors = torch.stack([torch.Tensor(i) for i in labels])
# transform = transforms.Compose([
    # # you can add other transformations in this list
    # transforms.ToTensor()
# ])
# transformed_data = transform(all_data)
# # data_set = torchvision.datasets.ImageFolder('training_data.npy' ,transform = transforms.ToTensor() )
data_set = torch.utils.data.TensorDataset(inputs_tensors,labels_tensors)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=3,shuffle=True, num_workers=2)
# training_data = all_data[:-500]lk
# testing_data = all_data[-500:]



if __name__ == '__main__':
 plot = visdomvisualize.Plot("Model A")
 plot.register_line("Loss", "Epoch", "Loss")
 for epoch in range(4):
  size_of_data = 0.0
  avgloss = 0.0
  runing_loss = 0.0
  for i,data in enumerate(data_loader , 0):
     size_of_data = size_of_data + 1
     inputs= data[0]
     # inputs = torch.LongTensor(inputs)
     labels= data[1]
     # labels = labels.long()
     labels = torch.FloatTensor(labels)
     optimizer.zero_grad()
     inputs = torch.unsqueeze(inputs, 1)
     # set_trace()
     outputs = AlexNet(inputs)
     loss = criterion(outputs , labels)
     print("Epoch:{0} , image:{1} , loss:{2} ".format(epoch , i , loss))
     loss.backward()
     optimizer.step()
     # plot.update_line("Loss", i + 1, loss.item())
  
     runing_loss +=loss.item()
  avgloss = runing_loss/size_of_data
  plot.update_line("Loss", epoch + 1, avgloss)
  print("saving checkpoint")
  torch.save({
            'epoch': epoch,
            'model_state_dict': AlexNet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, PATH)   
            
 print('finished')