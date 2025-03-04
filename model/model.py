import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=1024, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
            
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = MLP()
    print(model)