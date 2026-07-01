import scipy.stats as stats
import numpy as np

from utils_tools import find_kde, calculate_statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils_tools import SlidingWindowDataset
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

class convAE(torch.nn.Module):
    def __init__(self, args, conv_depth, lr = 2e-4, epochs = 100,confidence=0.99):
        super(convAE, self).__init__()
        self.args = args
        self.conv_depth = conv_depth
        self.lr = lr
        self.epochs = epochs
        self.confidence = confidence
        if conv_depth > 3:
            raise ValueError("Conv depth must small than 4") 
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        channels = [1] + [8 * (2 ** i) for i in range(conv_depth)]  # build channel sizes dynamically

        for i in range(1, conv_depth + 1):
            self.encoders.append(nn.Sequential(
                nn.Conv2d(channels[i-1], channels[i], kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ))

        # Decoder layers
        for i in range(conv_depth, 0, -1):
            output_padding = (1, 1)
            if i > 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=4, stride=2, padding=1, output_padding=output_padding),
                    nn.ReLU()
                ))
            else:
                self.decoders.append(
                    nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=4, stride=2, padding=1, output_padding=output_padding),
                )
        self.encoders.to(args.device)
        self.decoders.to(args.device)
        self.dropout = nn.Dropout(p=0.5)
        self.model = nn.Sequential(self.encoders,self.decoders)

    def forward(self, x):
        original_size = (x.size(2), x.size(3))
        odd_height = original_size[0] % 2 == 1
        odd_width = original_size[1] % 2 == 1

        # optional padding for odd dimensions
        if odd_height or odd_width:
            row_pad = 1 if odd_height else 0
            col_pad = 1 if odd_width else 0
            x = nn.functional.pad(x, (0, col_pad, 0, row_pad), 'replicate')

        # Encoding
        for encoder in self.encoders:
            x = encoder(x)
        x = self.dropout(x)
        # Decoding
        for decoder in self.decoders:
            x = decoder(x)

        x = x[:, :, :original_size[0], :original_size[1]]

        return x

    
    def fit(self,data,dataset_name):
        self.dataset_name = dataset_name
        self.sc = StandardScaler()
        data = self.sc.fit_transform(data)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = SlidingWindowDataset(tensor_data)
        self.trainloader = DataLoader(dataset, batch_size=32, shuffle=1)
        # import ipdb;ipdb.set_trace()
        note = 'convAE_layer{}_epoch{}_confidence{}_lr{}_{}'.format(self.conv_depth, self.epochs,self.confidence, self.lr, self.dataset_name)
        path = './checkpoint_bank'

        self.train()
        pass
    
    def train(self):
        path = './checkpoint_bank'
        note = 'convAE_layer{}_epoch{}_confidence{}_lr{}_{}'.format(self.conv_depth, self.epochs,self.confidence, self.lr, self.dataset_name)
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
                    imgs = imgs.cuda().float()
                    data_channel = imgs.unsqueeze(1)
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
            imgs = imgs.cuda().float()
            data_channel = imgs.unsqueeze(1)
            outputs = self.forward(data_channel)
            loss_list.append(torch.mean(loss_func_mse(outputs, data_channel),dim=(1, 2, 3)).detach().cpu().numpy())
        # import ipdb;ipdb.set_trace()
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

    def transform(self, data_original):
        raise NotImplementedError()


    def get_statistics(self, raw_test_data, is_train=False):
        data = self.sc.transform(raw_test_data)
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = SlidingWindowDataset(tensor_data)
        self.testloader = DataLoader(dataset, batch_size=32, shuffle=0)
        
        loss_list = []
        self.model.eval()
        for i, imgs in enumerate(self.testloader):
            imgs = imgs.cuda().float()
            data_channel = imgs.unsqueeze(1)
            outputs = self.forward(data_channel)
            loss_list.append(torch.mean(loss_func_mse(outputs, data_channel),dim=(1, 2, 3)).detach().cpu().numpy())
        
        T2 = np.hstack(loss_list)

        return [T2], [self.ctrl]
    
    def score(self,test_data,fault_start,fault_end,overall=False):
        T2_fault,_ = self.get_statistics(test_data)
        self.T2_fault = T2_fault
        T2_kzx = [self.ctrl]
        
        # adjust the time because of the window ↓ 
        fault_start -= (test_data.shape[0] - T2_fault[0].shape[0])
        fault_end -= (test_data.shape[0] - T2_fault[0].shape[0])
        TP = sum((np.array(T2_fault[0][fault_start:fault_end])>=T2_kzx[0]))
        TN = sum((np.array(T2_fault[0][:fault_start])<T2_kzx[0])) + \
                sum((np.array(T2_fault[0][fault_end:])<T2_kzx[0]))
        FP = sum((np.array(T2_fault[0][:fault_start])>=T2_kzx[0])) + \
                sum((np.array(T2_fault[0][fault_end:])>=T2_kzx[0]))
        FN = sum((np.array(T2_fault[0][fault_start:fault_end])<T2_kzx[0]))
        # Accuracy: fraction of correctly classified samples
        # import ipdb;ipdb.set_trace()
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
        plt.suptitle('convAE:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('T2')
        plt.axhline(self.ctrl, c='r')
        
        plt.savefig(path,dpi=300)
        plt.close()