#%% packages
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
os.getcwd()

# %% transform and load data
# TODO: set up image transforms
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]) 
# TODO: set up train and test datasets
trainset = torchvision.datasets.ImageFolder(root = 'CNN Image Classification/Multi Class Classification/train', 
                                            transform = transformer)
testset = torchvision.datasets.ImageFolder(root = 'CNN Image Classification/Multi Class Classification/test', 
                                            transform = transformer)
# TODO: set up data loaders
BATCH_SIZE = 6
trainloader = DataLoader(dataset = trainset, batch_size = BATCH_SIZE, shuffle = True)
testloader = DataLoader(dataset = testset, batch_size = BATCH_SIZE, shuffle = True)
# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3)       # [6, 6, 30, 30]
        self.pool = nn.MaxPool2d(2, 2)                      # [6, 6, 15, 15]
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3)      # [6, 16, 13, 13]
        self.fc1 = nn.Linear(16 * 6 * 6, 128)               # [6, 128]       
        self.fc2 = nn.Linear(128, 64)                       # [6, 64] 
        self.fc3 = nn.Linear(64, NUM_CLASSES)                         # [6, 3]      

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)       # Output: [6, 6, 30, 30]
        x = self.relu(x)
        x = self.pool(x)        # Output: [6, 6, 15, 15]
        x = self.conv2(x)       # Output: [6, 16, 13, 13]
        x = self.relu(x)
        x = self.pool(x)        # Output: [6, 16, 6, 6]
        x = torch.flatten(x, 1) # Output: [6, 16 * 6 * 6]
        x = self.fc1(x)         # Output: [6, 128]
        x = self.fc2(x)         # Output: [6, 64]
        x = self.fc3(x)         # Output: [6, 1]


# input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet()      
# model(input).shape

# %% loss function and optimizer
# TODO: set up loss function and optimizer
LR = 0.01
loss_fn = nn.CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        y_pred = model(inputs)

        loss = loss_fn(y_pred, labels)

        loss.backward()

        optimizer.step()
        
    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')


# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs,labels = data
    with torch.no_grad():
        outputs = model(inputs)

        predicted = torch.argmax(outputs, 1)
    
    y_test.extend(labels.numpy())
    y_test_hat.extend(predicted.numpy())

# %%
acc = accuracy_score(y_test, y_test_hat)
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_hat))
# %%
