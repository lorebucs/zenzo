import torch
import torch.nn as nn

class SimplestNN(nn.Module):
    def __init__(self):
        super(SimplestNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x
    
class LogisticClassifier(nn.Module):
    def __init__(self):
        super(LogisticClassifier, self).__init__()
        self.lin = nn.Linear(2, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.lin(x))
        return x