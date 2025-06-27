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
import seaborn as sns

# %% Define Transformer
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# %% Define Datasets and DataLoaders
train_data = torchvision.datasets.ImageFolder('train', transform = transformer)
test_data = torchvision.datasets.ImageFolder('test', transform = transformer)

BATCH_SIZE = 4

trainloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
testloader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = True)
# %% Define Neural Network Class
classes = ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(classes)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(16 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = self.conv1(x)           # Output: BS, 6, 100, 100
        x = self.relu(x)
        x = self.pool(x)            # Output: BS, 6, 50, 50
        x = self.conv2(x)           # Output: BS, 16, 50, 50
        x = self.relu(x)
        x = self.pool(x)            # Output: BS, 16, 25, 25
        x = torch.flatten(x, 1)     # Output: BS, 16 * 25 * 25
        x = self.fc1(x)             # Output: BS, 128      
        x = self.relu(x) 
        x = self.dropout(x)           
        x = self.fc2(x)             # Output: BS, 64
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)             # Output: BS, NUM_CLASSES

        return x

# %% Define loss function and optimizer
LR = 0.001

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

# %% Training Loop
NUM_EPOCHS = 100
losses = []

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

    avg_loss = avg_loss/ len(trainloader)
    losses.append(avg_loss)

    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')

#%% Visualizing Results
sns.lineplot(x = range(NUM_EPOCHS), y = losses)

# %% Model Evlauation
model.eval()

with torch.no_grad():
    y_test = []
    y_test_hat = []

    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        outputs = model(inputs)

        predicted = torch.argmax(outputs, 1)

        y_test.extend(labels.numpy())

        y_test_hat.extend(predicted.numpy())

# %% Model Accuracy
accuracy = accuracy_score(y_test, y_test_hat)
print(f' Model Accuracy: {accuracy * 100:.2f}%')

#%% Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_hat)
sns.heatmap(conf_matrix, annot = True, xticklabels = classes, yticklabels = classes)
# %%
