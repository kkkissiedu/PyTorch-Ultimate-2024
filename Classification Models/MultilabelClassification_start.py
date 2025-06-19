#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
mldataset = MultilabelDataset(X = X_train, y = y_train)

# TODO: create train loader
dloader = DataLoader(mldataset, batch_size = 32, shuffle = True)

# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.a1(x)
        x = self.fc2(x)
        return x


# TODO: define input and output dim
input_dim = mldataset.X.shape[1]
hidden_dim = 20
output_dim = mldataset.y.shape[1]

# TODO: create a model instance
model = MultiLabelClassifier(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim)

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()

LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    avg_loss = 0
    for j, (X, y) in enumerate(dloader):
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fn(y_pred, y)
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        avg_loss += loss.item()

    avg_loss = avg_loss / len(dloader)
    losses.append(avg_loss)   
    # TODO: print epoch and loss at end of every 10th epoch
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: Loss = {avg_loss}')
    
# %% losses
# TODO: plot losses
sns.lineplot(x = range(number_epochs), y = losses)
# %% test the model
# TODO: predict on test set
model.eval()
with torch.no_grad():
    y_test_logits = model(X_test)

    y_test_probs = torch.sigmoid(y_test_logits)

    y_pred_binary = (y_test_probs > 0.5).int()

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.numpy()]

# TODO: get most common class count
most_common_cnt = Counter(y_test_str).most_common()[0][1]

# TODO: print naive classifier accuracy
print(f'Naive Classifier Accuracy: {most_common_cnt/len(y_test_str)*100}%')

# %% Test accuracy
# TODO: get test set accuracy
model_accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Model Test Accuracy: {model_accuracy:.4f}')
# %%
