import scipy.stats as stats
import numpy as np

from utils_tools import find_kde, calculate_statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils_tools import NoWindowDataset
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

    
loss_func_mse = torch.nn.MSELoss(reduction='none')

class MLPAE(torch.nn.Module):
    def __init__(self, args, num_layers, growth_rate, lr = 2e-4, epochs = 100,confidence=0.99):
        super(MLPAE, self).__init__()
        self.args = args
        self.lr = lr
        self.epochs = epochs
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.confidence = confidence
        self.dropout = nn.Dropout(p=0.5)

    def _construct_layers(self, input_dim):
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        layer_dims = [input_dim]
        for _ in range(self.num_layers - 1):  # Subtracting 1 implies factoring for the output layer
            layer_dims.append(int(layer_dims[-1] * self.growth_rate))

        for i in range(len(layer_dims)-1):
            self.encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.encoder_layers.append(nn.ReLU())
            self.decoder_layers.insert(0, nn.ReLU())
            self.decoder_layers.insert(0, nn.Linear(layer_dims[i+1], layer_dims[i]))

        self.decoder_layers = self.decoder_layers[:-1]  # Remove ReLU for the final layer
        self.model = nn.Sequential(self.encoder_layers,self.decoder_layers)
        self.model.to(self.args.device)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x
    
    def fit(self,data,dataset_name):
        self.dataset_name = dataset_name
        # import ipdb;ipdb.set_trace()
        self._construct_layers(data.shape[-1])
        self.sc = StandardScaler()
        data = self.sc.fit_transform(data)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = NoWindowDataset(tensor_data)
        self.trainloader = DataLoader(dataset, batch_size=32, shuffle=1)
        self.train()
        pass
    
    def train(self):
        note = 'AE_layer{}_wide{}_epoch{}_confidence{}_lr{}_{}'.format(self.num_layers, self.growth_rate, self.epochs,self.confidence, self.lr, self.dataset_name)
        path = './checkpoint_bank'
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.epochs)
        if os.path.exists(os.path.join(path, note+'.pt')):
            # Load checkpoint if .pt file exists
            print('Model checkpoint exists, load it.')
            check_dict = torch.load(os.path.join(path, note+'.pt'))
            self.model.load_state_dict(check_dict['model'])
        else:
            self.model.train()
            for epoch in range(self.epochs):
                for i, imgs in enumerate(self.trainloader):
                    # import ipdb;ipdb.set_trace()
                    data_channel = imgs.cuda().float()
                    outputs = self.forward(data_channel)
                    optimizer.zero_grad()
                    loss = torch.mean(loss_func_mse(outputs, data_channel))
                    
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if epoch % 5 == 0:
                    print('loss',loss)

        loss_list = []
        self.model.eval()
        for i, imgs in enumerate(self.trainloader):
            data_channel = imgs.cuda().float()
            outputs = self.forward(data_channel)
            loss_list.append(torch.mean(loss_func_mse(outputs, data_channel),1).detach().cpu().numpy())
        
        loss_array = np.hstack(loss_list)
        self.ctrl = find_kde(loss_array,self.confidence)

        check_dict = {'args':self.args, 'epochs':self.epochs, 'model':self.model.state_dict()}
        if optimizer is not None:
            check_dict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            check_dict['shceduler'] = scheduler.state_dict()

        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(check_dict, os.path.join(path, note+'.pt'))
        
        # import ipdb;ipdb.set_trace()

    def transform(self, data_original):
        raise NotImplementedError()


    def get_statistics(self, raw_test_data, is_train=False):
        data = self.sc.transform(raw_test_data)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = NoWindowDataset(tensor_data)
        self.testloader = DataLoader(dataset, batch_size=32, shuffle=0)
        
        loss_list = []
        self.model.eval()
        for i, imgs in enumerate(self.testloader):
            data_channel = imgs.cuda().float()
            outputs = self.forward(data_channel)
            loss_list.append(torch.mean(loss_func_mse(outputs, data_channel),1).detach().cpu().numpy())
        
        T2 = np.hstack(loss_list)

        return [T2], [self.ctrl]
    
    def score(self,test_data,fault_start,fault_end,overall=False):
        T2_fault,_ = self.get_statistics(test_data)
        self.T2_fault = T2_fault
        T2_kzx = [self.ctrl]
        # import ipdb;ipdb.set_trace()
        TP = sum((np.array(T2_fault[0][fault_start:fault_end])>=T2_kzx[0]))
        TN = sum((np.array(T2_fault[0][:fault_start])<T2_kzx[0])) + \
                sum((np.array(T2_fault[0][fault_end:])<T2_kzx[0]))
        FP = sum((np.array(T2_fault[0][:fault_start])>=T2_kzx[0])) + \
                sum((np.array(T2_fault[0][fault_end:])>=T2_kzx[0]))
        FN = sum((np.array(T2_fault[0][fault_start:fault_end])<T2_kzx[0]))
        # Accuracy: fraction of correctly classified samples
        if overall:
            # Precision= TP / TP + FP
            # Recall = TP / TP + FN
            # F1 = 2 P * R / P + R
            Pre = TP/(TP+FP)
            Rec = TP/(TP+FN)
            return  2*Pre*Rec/(Pre+Rec)
        else:
            return TP/(fault_end-fault_start)

    def plot(self,path):
        plt.suptitle('MLPAE:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('T2')
        plt.axhline(self.ctrl, c='r')
        
        plt.savefig(path,dpi=300)
        plt.close()