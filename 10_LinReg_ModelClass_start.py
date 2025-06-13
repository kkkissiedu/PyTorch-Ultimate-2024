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

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
    


input_dim = 1
output_dim= 1

model = LinearRegressionTorch(input_dim, output_dim)


# %% Loss Function
loss_fun = nn.MSELoss()
# %% Optimizer
lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

# %% Training Loop
losses, slope, bias = [], [], []

epochs = 1000

for epoch in range(epochs):
    # Set gradients to zero
    optimizer.zero_grad()

    # Forward Pass
    y_pred = model(X)

    # Compute Loss
    loss = loss_fun(y_pred, y_true)

    #Backward Pass (Compute gradients)
    loss.backward()

    # Update Weights
    optimizer.step()

    # Get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # Store loss
    losses.append(float(loss.data))

    # Print loss
    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.data:.4f}')

# %% Visualize model training
sns.scatterplot(x = range(epochs), y = losses)

# %% Visualize the bias development
sns.scatterplot(x = range(epochs), y = bias)

# %% Visualize the slope development
sns.scatterplot(x = range(epochs), y = slope)

# %% Visualize the final model
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x = X_list, y = y_list)
sns.lineplot(x = X_list, y = y_pred, color = 'red')

# %%
