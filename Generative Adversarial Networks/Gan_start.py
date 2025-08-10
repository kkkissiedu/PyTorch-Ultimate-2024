#%% packages
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform

import seaborn as sns
sns.set_theme(rc={'figure.figsize':(12,12)})
#%% create training data
TRAIN_DATA_COUNT = 1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)]) # np.linspace(0, 2 * np.pi, 100)
# Generating x and y data
x = 16 * ( np.sin(theta) ** 3 )
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
sns.scatterplot(x=x, y=y)

#%% prepare tensors and dataloader
train_data = torch.Tensor(np.stack((x, y), axis=1))

train_labels = torch.zeros(TRAIN_DATA_COUNT)
train_set = [
    (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
]

#  dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
#%% initialize discriminator and generator
discriminator = nn.Sequential(
    nn.Linear(2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

generator = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# %% training
LR = 0.001
NUM_EPOCHS = 2000
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = LR)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr = LR)

for epoch in tqdm(range(NUM_EPOCHS), desc="Training Progress"):

    epoch_loss_d = 0.0
    epoch_loss_g = 0.0

    for n, (real_samples, _) in enumerate(train_loader):
        
        if real_samples.shape[0] != BATCH_SIZE:

            continue
        
        # Training Discriminator
        discriminator.zero_grad()

        real_samples_labels = torch.ones((BATCH_SIZE, 1))

        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((BATCH_SIZE, 1))

        all_samples = torch.cat((real_samples, generated_samples.detach()))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)

        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Training Generator
        generator.zero_grad()

        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        generated_samples = generator(latent_space_samples)

        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        
        loss_generator.backward()
        optimizer_generator.step()

        epoch_loss_d += loss_discriminator.item()
        epoch_loss_g += loss_generator.item()

        # Show progress
        if epoch % 10 == 0 and epoch > 0:
            # Average the losses over all batches
            avg_loss_d = epoch_loss_d / len(train_loader)
            avg_loss_g = epoch_loss_g / len(train_loader)

            tqdm.write(f"Epoch: {epoch} | D Loss: {loss_discriminator.item():.4f} | G Loss: {loss_generator.item():.4f}")
            
            with torch.no_grad():
                latent_space_samples = torch.randn(1000, 2)
                generated_samples = generator(latent_space_samples).detach()
            
            
            plt.figure()
            plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
            plt.plot(x, y, ".", color='red', alpha=0.1, label='Real Data')
            plt.xlim((-20, 20))
            plt.ylim((-20, 15))
            plt.title(f"Epoch {epoch}")
            plt.legend()
            plt.savefig(f"train_progress/image{str(epoch).zfill(4)}.jpg")
            plt.show()
            plt.close()


    

# %% check the result
latent_space_samples = torch.randn(10000, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.text(10, 15, f"Epoch {epoch}")

# %%