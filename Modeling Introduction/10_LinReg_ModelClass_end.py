#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

# %% Linear Regression Model Class
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.Linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.Linear(x)
#%% Model Instantiation
input_dim = 1
output_dim = 1
lr = 0.01

model = LinearRegression(input_dim, output_dim)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

#%%Training Loop
epoch_losses, slope, bias = [], [], []
epochs = 250
batch_size = 2

for epoch in range(epochs):
    # Set loss = 0 for each epoch
    current_loss = 0

    for i in range(0, X.shape[0], batch_size):
        # Set gradients to zero
        optimizer.zero_grad()

        # Forward Pass
        y_pred = model(X[i:i + batch_size])
        loss = loss_func(y_pred, y_true[i:i + batch_size])

        # Backward Pass
        loss.backward()

        #Update Weights
        optimizer.step()

        current_loss += loss.item() 

    # Calculate and store the average loss per epoch 
    avg_loss = current_loss / (X.shape[0] / batch_size)
    
    #Record Loss per Epoch
    epoch_losses.append(avg_loss)

    #Record Slope and Bias
    slope.append(model.Linear.weight.item())
    bias.append(model.Linear.bias.item())

    #Plot Loss for every 100 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Average loss: {avg_loss:.4f}, slope = {model.Linear.weight.item():.4f}, bias = {model.Linear.bias.item():4f}')
#%% Plot Loss 
ax = sns.scatterplot(x = range(epochs), y = epoch_losses, label = 'Loss againt Epoch')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

#%% Plot Slope and Bias
sns.lineplot(x = range(epochs), y = slope, label = 'Slope against Epoch')
sns.lineplot(x = range(epochs), y = bias, label = 'Bias against Epoch')

# %% Visualize the final model
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x = X_list, y = y_list)
sns.lineplot(x = X_list, y = y_pred, color = 'red')