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
from sklearn.metrics import accuracy_score
os.getcwd()

#%% transform, load data
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root = 'CNN Image Classification/Binary Clasification/data/train', 
                                            transform = transform)
testset = torchvision.datasets.ImageFolder(root = 'CNN Image Classification/Binary Clasification/data/test', 
                                            transform = transform)

trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size = batch_size, shuffle = True)

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter._next_data()
imshow(torchvision.utils.make_grid(images, nrow=2))


# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3)   # Out: BS, 6, 30, 30 
        self.pool = nn.MaxPool2d(2, 2)                  # Out: BS, 6, 15, 15
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3)  # Out: BS, 16, 13, 13
        self.fc1 = nn.Linear(16 * 6 * 6, 128)           # After next pool: BS, 16, 6, 6
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)           # Out: BS, 6, 30, 30
        x = self.relu(x)               
        x = self.pool(x)            # Out: BS, 6, 15, 15   
        x = self.conv2(x)           # Out: BS, 16, 13, 13  
        x = self.relu(x)
        x = self.pool(x)            # Out: BS, 16, 6, 6
        x = torch.flatten(x, 1)     # Out: BS, 16*6*6
        x = self.fc1(x)             # Out: BS, 128
        x = self.relu(x)
        x = self.fc2(x)             # Out: BS, 64
        x = self.relu(x)
        x = self.fc3(x)             # Out: BS, 1
        x = self.sigmoid(x)

        return x 
    

#%% init model
model = ImageClassificationNet()      
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.8)


# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # Reset gradients
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(inputs) 

        # calc losses
        loss = loss_fn(y_pred, labels.reshape(-1, 1).float())
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)},'
                  f'Loss: {loss.item():.4f}')
# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.