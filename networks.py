import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, dim=50, num_classes=3, calibration_paradigm="TopLabel"):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, num_classes) 
        
        if calibration_paradigm == "TopLabel":
            self.classifier2 = nn.Linear(dim, 2)
        else:
            self.classifier2 = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 
        return x
    
    def forward_classifier2(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.classifier2(x) 
        return x

