#%% Importing Packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# %% Define Transformer
transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])
# %% Define Datasets and DataLoaders
train_data = torchvision.datasets.ImageFolder('train', transform = transformer)
test_data = torchvision.datasets.ImageFolder('test', transform = transformer)

BATCH_SIZE = 6

trainloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
testloader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = True)
# %% Define Neural Network Class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)           # Output: BS, 6, 62, 62
        x = self.relu(x)
        x = self.pool(x)            # Output: BS, 6, 31, 31
        x = self.conv2(x)           # Output: BS, 16, 29, 29
        x = self.relu(x)
        x = self.pool(x)            # Output: BS, 16, 14, 14
        x = torch.flatten(x, 1)     # Output: BS, 16 * 14 * 14
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# %% Define loss function and optimizer
LR = 0.001

model = CNN()
loss_fn = nn.BCEWithLogitsLoss
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

# %% Training Loop
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    avg_loss = 0

    for i, data in enumerate(trainloader, 0):

        input, labels = data

        # Reset Gradients
        optimizer.zero_grad()

        # Forward Pass
        preds = model(input)

        # Loss Calculation
        loss = loss_fn(preds, labels)
        avg_loss += loss.item()

        # Backward Pass
        loss.backward()
        
        # Update Gradients
        optimizer.step()

    avg_loss = avg_loss/BATCH_SIZE

    print(f'Epoch: {epoch}, Loss: {avg_loss}')
    

    


