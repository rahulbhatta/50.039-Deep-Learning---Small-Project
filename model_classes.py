import torch
import torch.nn as nn
import torch.nn.functional as F

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # Conv2D: 1 input channel, 4 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(16, 128, 3, 1)
        self.pool1 = nn.MaxPool2d(3)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(128, 256, 2, 1)
        self.pool2 = nn.AvgPool2d(3)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(57600, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.pool2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        #print("After flatten:", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # Conv2D: 1 input channel, 4 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(16, 128, 3, 1)
        self.pool1 = nn.MaxPool2d(3)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(128, 256, 2, 1)
        self.pool2 = nn.AvgPool2d(3)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(57600, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.pool2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        #print("After flatten:", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output