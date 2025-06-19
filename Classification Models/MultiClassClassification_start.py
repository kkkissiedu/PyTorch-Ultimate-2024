#%% packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# %% data import
iris = load_iris()
X = iris.data
y = iris.target


df = pd.DataFrame(data=X, columns=iris.feature_names)

df['species'] = y

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species_name', s=100)
plt.title('Sepal Length vs. Sepal Width by Species')
plt.show()

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 

# %% convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% dataset
class MultiClassDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# %%
iris_data = MultiClassDataset(X = X_train, y = y_train)

# %% dataloader
dloader = DataLoader(iris_data, batch_size = 32, shuffle = True)
# %% check dims
print(f'X shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}')
# %% define class
class MultiClassModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.L1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.S1 = nn.ReLU()
        self.L2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)

    def forward(self, x):
        x = self.L1(x)
        x = self.S1(x)
        x = self.L2(x)

        return x

# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(np.unique(iris_data.y))
# %% create model instance
model = MultiClassModel(NUM_FEATURES = NUM_FEATURES, NUM_CLASSES = NUM_CLASSES, HIDDEN_FEATURES = HIDDEN)
# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
# %% training
NUM_EPOCHS = 100  
losses = []

for epoch in range(NUM_EPOCHS):
    avg_loss = 0.0
    for X, y in dloader:
        
        # Initialize Gradients and Average Loss
        optimizer.zero_grad()
        
        # Forward Pass
        y_pred_log = model(X)

        # Loss Calculation
        loss = criterion(y_pred_log, y.long())

        # Backward Pass
        loss.backward()

        # Update Weights
        optimizer.step()

        avg_loss += loss.item() 

    avg_loss = avg_loss/len(dloader)  
    losses.append(avg_loss)     

    print(f'Epoch {epoch + 1}: Loss: {avg_loss}')

# %% show losses over epochs
sns.lineplot(x = range(NUM_EPOCHS), y = losses)

# %% test the model
X_test_torch = torch.from_numpy(X_test)

with torch.no_grad():
    y_test_log = model(X_test_torch)
    y_test_pred = torch.max(y_test_log, 1)


# %% Accuracy
accuracy_score(y_test, y_test_pred.indices)

# %%
from collections import Counter
most_common_cnt = Counter(y_test).most_common()[0][1]
print(f'Naive Classifier Accuracy: {(most_common_cnt/len(y_test) * 100):1f}%')
# %%
