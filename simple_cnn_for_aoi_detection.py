import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FC = nn.MSELoss()
LEARNING_RATE = 1e-4
EPOCH = 100
BATCH_SIZE = 2**7

### plot2array
def plot2array(loss_list, max_cnt):
    plt.figure(figsize=(10,7))
    plt.plot(loss_list, color = 'g', linewidth = 1)
    plt.xlim([0,max_cnt])
    fig = plt.gcf()
    fig.canvas.draw()
    plot_array = np.array(fig.canvas.renderer._renderer)
    plt.close('all')
    return plot_array



### forge dataset
class aoi_dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        label_file = pd.read_csv(base_dir + '/labels.csv')
        self.file_nms = label_file.file_nm
        self.ys = label_file.iloc[:,1:].values

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = cv2.imread(self.base_dir + f'/imgs/{self.file_nms[idx]}')
        img_len = x.shape[0]
        x = x.reshape(3, img_len, img_len) # image is square shape (length == height)
        self.x_data = torch.from_numpy(x).float()

        y = self.ys[idx]
        y = np.array([(img_len - i)/img_len for i in y])
        self.y_data = torch.from_numpy(y).float()
        return self.x_data, self.y_data

data_dir = 'C:/ALL/vision_reader_dev/OCR/OCR_DATA/augmented_data'
aoi_data = aoi_dataset(data_dir)

temp_dataset = DataLoader(dataset = aoi_data, batch_size = BATCH_SIZE, shuffle = True)

len(temp_dataset)

### model
#class simpleCNN(nn.Module):
#    def __init__(self): # in 480
#        super().__init__()
#        self.conv1 = nn.Conv2d(3, 1, 11, stried=3, padding = 1) # out 158
#        self.batch_norm = nn.BatchNorm1d()
#        self.pool1 = nn.MaxPool2d(2,2) # out 158/2 = 79
#        self.conv2 = nn.Conv2d(1, 1, 7, stride = 2, padding = 1) # out = 38
#        self.pool2 = nn.MaxPool2d(2,2) # out 19
#        self.ffn1 = nn.Linear(1521, 2**7)
#        self.ffn2 = nn.Linear(2**7, 2**5)
#        self.out = nn.Linear(2**5, 2**3)
#    
#    def forward(self, x):
#        x = self.pool1(F.relu(self.conv1(x)))
#        x = self.pool2(F.relu(self.conv2(x)))
#        x = torch.flatten(x,1)
#        x = F.relu(self.ffn1(x))
#        x = F.relu(self.ffn2(x))
#        x = self.out(x)
#        return x

# model
base_channel = 3
class simpleCNN(nn.Module):
    def __init__(self): # in 480
        super().__init__()
        self.conv1 = nn.Conv2d(base_channel, base_channel*2, 7, stride=1, padding=3)
        self.bnorm1 = nn.BatchNorm2d(base_channel*2) # 32
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(base_channel*2, base_channel*4, 7, stride=1, padding=3)
        self.bnorm2 = nn.BatchNorm2d(base_channel*4) # 64
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(base_channel*4, base_channel*6, 3, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(base_channel*6) #128
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(base_channel*6, base_channel*4, 3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm2d(base_channel*4) # 256
        self.pool4 = nn.MaxPool2d(2,2) 

        self.conv5 = nn.Conv2d(base_channel*4, base_channel*2, 3, stride=1, padding=1)
        self.bnorm5 = nn.BatchNorm2d(base_channel*2) # 256
        self.pool5 = nn.MaxPool2d(2,2)

        self.conv6 = nn.Conv2d(base_channel*2, base_channel, 3, stride=1, padding=1)
        self.bnorm6 = nn.BatchNorm2d(base_channel) # 256
        self.pool6 = nn.MaxPool2d(2,2) 

        self.linear1 = nn.Linear(3*7*7, 2**7)
        self.linear2 = nn.Linear(2**7, 2**6)
        self.linear3 = nn.Linear(2**6, 2**5)
        self.linear4 = nn.Linear(2**5, 2**4)
        self.out = nn.Linear(2**4, 2**3)


    def forward(self, x):
        x = self.pool1(F.relu(self.bnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.bnorm2(self.conv2(x))))
        x = self.pool3(F.relu(self.bnorm3(self.conv3(x))))
        x = self.pool4(F.relu(self.bnorm4(self.conv4(x))))
        x = self.pool5(F.relu(self.bnorm5(self.conv5(x))))
        x = self.pool6(F.relu(self.bnorm6(self.conv6(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.out(x)
        return x

### train
model = simpleCNN().to(DEVICE)
next(model.parameters()).is_cuda


#optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=0.01)

loss_by_batch = list()
loss_at_end_of_every_epoch = list()
loss_holder = list()
for epoch in range(EPOCH):
    running_loss = 0.0
    temp_list = list()
    for i, data in enumerate(temp_dataset, 1):
        inputs, labels = data
        inputs = inputs/255 # 0~1 norm
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = LOSS_FC(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f'{epoch + 1}-{i}, mean_loss : {running_loss/i}')
        
        loss_holder.append(running_loss/i)
        temp_list.append(running_loss/i)
        running_loss = 0.0
            
        temp_array = plot2array(loss_holder, len(temp_dataset)*EPOCH)
        cv2.imshow('loss',temp_array)
        cv2.waitKey(1)
    loss_at_end_of_every_epoch.append(running_loss/i)
    loss_by_batch.append(temp_list)


temp_dict = dict()
for i in range(len(loss_by_batch)):
    temp_dict[i] = loss_by_batch[i]

pd.DataFrame.from_dict(temp_dict).to_csv("~/ALL/test.csv")