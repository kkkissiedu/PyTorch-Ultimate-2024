#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# %% import image
img = Image.open('CNN Image Classification/kiki.jpg')
img

# %% compose a series of steps
preprocess_steps = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomRotation(50),
    transforms.CenterCrop(200),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean = [0.5], std = [0.5])
])
x = preprocess_steps(img)
x.shape
# %% Get the Mean and Standard Deviation of an image
x.mean(), x.std() 
