import torch.nn as nn

class DummyChemModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.layer(x)