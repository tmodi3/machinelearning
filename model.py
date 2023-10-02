import torch
import torch.nn as nn

class ReceiptsPredictor(nn.Module):
    def __init__(self):
        super(ReceiptsPredictor, self).__init__()
        self.layer = nn.Linear(1, 1)  # Simple linear layer

    def forward(self, x):
        return self.layer(x)
