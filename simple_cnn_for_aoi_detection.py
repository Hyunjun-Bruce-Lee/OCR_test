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

### check gpu
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


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

temp_dataset = DataLoader(dataset = aoi_data, batch_size = 2**5, shuffle = True)



### model
class simpleCNN(nn.Module):
    def __init__(self): # in 480
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 11, stride=3, padding = 1) # out 158
        self.pool1 = nn.MaxPool2d(2,2) # out 158/2 = 79
        self.conv2 = nn.Conv2d(1, 1, 7, stride = 2, padding = 1) # out = 38
        self.pool2 = nn.MaxPool2d(2,2) 
        self.ffn1 = nn.Linear(1521, 2**7)
        self.ffn2 = nn.Linear(2**7, 2**5)
        self.out = nn.Linear(2**5, 2**3)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.ffn1(x))
        x = F.relu(self.ffn2(x))
        x = self.out(x)
        return x


### train
model = simpleCNN().to(DEVICE)
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(temp_dataset, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i != 0:
            print(f'{epoch + 1}, mean_loss : {running_loss/i}')
            running_loss = 0.0




### predict
label_df = pd.read_csv('C:/ALL/vision_reader_dev/OCR/OCR_DATA/augmented_data/labels.csv')

idx = 0
test_img = cv2.imread(f'C:/ALL/vision_reader_dev/OCR/OCR_DATA/augmented_data/imgs/{label_df.loc[0,"file_nm"]}')
test_img = test_img.reshape(3,test_img.shape[0], test_img.shape[1])
test_img = np.expand_dims(test_img, 0)
test_img = torch.from_numpy(test_img).float()
test_img = test_img.to(DEVICE)
act_label = label_df.iloc[idx,1:].values


model(test_img)