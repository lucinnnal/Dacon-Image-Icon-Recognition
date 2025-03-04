import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import MLP
from utils import ImageDataset

# Dataset & DataLoader
train_csv_path = "./train.csv"
test_csv_path = "./test.csv"

train_dataset = ImageDataset(train_csv_path)
test_dataset = ImageDataset(test_csv_path)

train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

input, target = next(iter(train_dataloader))

print(f"batch input shape: {input.shape}")
print(f"batch output shape: {target.shape}")

image_data, label_idx = train_dataset[0]
    
print(f"image data shape : {image_data.shape}")
print(f"label data : {label_idx}")

# Model, Criterion, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = MLP()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)