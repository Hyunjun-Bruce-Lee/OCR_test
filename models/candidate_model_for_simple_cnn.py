import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from torchvision.io import read_image

import torchsummary


# model1 -> loss converges, but loss spikes every epoch
class simpleCNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 11, stride=3, padding = 1) 
        self.batch_norm = nn.BatchNorm1d(1)
        self.pool1 = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(1, 1, 7, stride = 2, padding = 1)
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
    

# model1-1
class simpleCNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 11, stride=3, padding = 1) 
        self.batch_norm = nn.BatchNorm1d(1)
        self.pool1 = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(1, 1, 7, stride = 2, padding = 1)
        self.pool2 = nn.MaxPool2d(2,2) 

        self.ffn_lt_1 = nn.Linear(1521, 2**7)
        self.ffn_lt_2 = nn.Linear(2**7, 2**5)
        self.ffn_lt_3 = nn.Linear(2**5, 2**3)
        self.ffn_lt_out = nn.Linear(2**3, 2)

        self.ffn_rt_1 = nn.Linear(1521, 2**7)
        self.ffn_rt_2 = nn.Linear(2**7, 2**5)
        self.ffn_rt_3 = nn.Linear(2**5, 2**3)
        self.ffn_rt_out = nn.Linear(2**3, 2)

        self.ffn_lb_1 = nn.Linear(1521, 2**7)
        self.ffn_lb_2 = nn.Linear(2**7, 2**5)
        self.ffn_lb_3 = nn.Linear(2**5, 2**3)
        self.ffn_lb_out = nn.Linear(2**3, 2)

        self.ffn_rb_1 = nn.Linear(1521, 2**7)
        self.ffn_rb_2 = nn.Linear(2**7, 2**5)
        self.ffn_rb_3 = nn.Linear(2**5, 2**3)
        self.ffn_rb_out = nn.Linear(2**3, 2)
        
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)

        lt = F.relu(self.ffn_lt_1(x))
        lt = F.relu(self.ffn_lt_2(lt))
        lt = F.relu(self.ffn_lt_3(lt))
        lt = F.relu(self.ffn_lt_out(lt))

        rt = F.relu(self.ffn_rt_1(x))
        rt = F.relu(self.ffn_rt_2(rt))
        rt = F.relu(self.ffn_rt_3(rt))
        rt = F.relu(self.ffn_rt_out(rt))

        lb = F.relu(self.ffn_lb_1(x))
        lb = F.relu(self.ffn_lb_2(lb))
        lb = F.relu(self.ffn_lb_3(lb))
        lb = F.relu(self.ffn_lb_out(lb))

        rb = F.relu(self.ffn_rb_1(x))
        rb = F.relu(self.ffn_rb_2(rb))
        rb = F.relu(self.ffn_rb_3(rb))
        rb = F.relu(self.ffn_rb_out(rb))

        out = torch.cat((lt,rt,lb,rb),1)
        return out

    


# model2 -> -> loss converges, but loss spikes every epoch (performance little better then model 1)
base_channel = 3
class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(base_channel, base_channel*2, 7, stride=1, padding=3)
        self.bnorm1 = nn.BatchNorm2d(base_channel*2) 
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(base_channel*2, base_channel*4, 7, stride=1, padding=3)
        self.bnorm2 = nn.BatchNorm2d(base_channel*4) 
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(base_channel*4, base_channel*6, 3, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm2d(base_channel*6) 
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(base_channel*6, base_channel*4, 3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm2d(base_channel*4) 
        self.pool4 = nn.MaxPool2d(2,2) 

        self.conv5 = nn.Conv2d(base_channel*4, base_channel*2, 3, stride=1, padding=1)
        self.bnorm5 = nn.BatchNorm2d(base_channel*2)
        self.pool5 = nn.MaxPool2d(2,2)

        self.conv6 = nn.Conv2d(base_channel*2, base_channel, 3, stride=1, padding=1)
        self.bnorm6 = nn.BatchNorm2d(base_channel)
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



# model3 ->
base_channel = 3
class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(base_channel, base_channel*2, 7, stride=1, padding=3)
        self.conv1_2 = nn.Conv2d(base_channel*2, base_channel*4, 5, stride=1, padding = 3)
        self.bnorm1 = nn.BatchNorm2d(base_channel*4) 
        self.max_p = nn.MaxPool2d(5)
        self.conv2_1 = nn.Conv2d(base_channel*4, base_channel*2, 5, stride=1, padding = 1)
        self.conv2_2 = nn.Conv2d(base_channel*2, base_channel, 3, stride=1)
        self.bnorm2 = nn.BatchNorm2d(base_channel)
        self.avg_p = nn.AvgPool2d(5)
    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.bnorm1(self.conv1_2(x)))
        x = self.max_p(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.bnorm2(self.conv2_2(x)))
        x = self.avg_p(x)
        return x

model = simpleCNN()
torchsummary.summary(model, (3,960,960))

