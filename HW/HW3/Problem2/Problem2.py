"""
Code written by Joey Wilson, 2023.
"""

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Transformation for training set
# See https://pytorch.org/vision/stable/transforms.html
def transform_train():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32), antialias=True),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return transform

# Transformation for test set
def transform_test():
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32), antialias=True),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return transform


# Car dataset
class Cars(Dataset):
  def __init__(self, root, transform=None, device="cpu"):
    self.root = root # Location of files
    self.image_labels = np.genfromtxt(os.path.join(self.root, 'Labels.txt'))
    self.image_files = []
    for i in range(self.image_labels.shape[0]):
      image_file = os.path.join(self.root, str(i)+".png")
      self.image_files.append(image_file)
    self.transform=transform # Your transform
    self.device = device
  
  # Returns the length of an epoch
  def __len__(self):
    return len(self.image_files)

  # Get an individual training item by index
  def __getitem__(self, index):
    # Obtain the file name corresponding to index
    file_name = self.image_files[index]
    # Open the image
    img = Image.open(file_name)
    # Convert the image to a numpy array
    img_mat = np.copy(np.asarray(img)[:, :, :3])
    # Obtain the image classification ground truth
    file_label = int(self.image_labels[index])
    return self.transform(img_mat), file_label

  # Combine data examples into a batch
  def collate_fn(self, data):
    B = len(data)
    img_batch = torch.stack([data[i][0] for i in range(B)])
    label_batch = torch.tensor([data[i][1] for i in range(B)])
    return img_batch, label_batch


# TODO: Go through PyTorch image classification tutorial
class TutorialNet(nn.Module):
  def __init__(self):
    super().__init__()
    # This is where we define layers
    self.net = nn.Sequential(
      nn.Conv2d(3, 6, 5),          # Conv layer: in_channels=3, out_channels=6, kernel_size=5
      nn.ReLU(),                   # Activation function
      nn.MaxPool2d(2, 2),          # Max pooling: kernel_size=2, stride=2
      nn.Conv2d(6, 16, 5),         # Conv layer: in_channels=6, out_channels=16, kernel_size=5
      nn.ReLU(),                   # Activation function
      nn.MaxPool2d(2, 2),          # Max pooling
      nn.Flatten(),                # Flatten the tensor for the fully connected layers
      nn.Linear(16 * 5 * 5, 120),  # Fully connected layer
      nn.ReLU(),                   # Activation function
      nn.Linear(120, 84),          # Fully connected layer
      nn.ReLU(),                   # Activation function
      nn.Linear(84, 3)             # Output layer: output size adjusted to 3 classes
    )

  def forward(self, x):
    # This is where we pass the input through our network
    x = self.net(x)
    return x


# TODO: Create your own network
class YourNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      # First convolutional block
      nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels: 3, output channels: 32
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),  # Output: 32 x 16 x 16

      # Second convolutional block
      nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output channels: 64
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),  # Output: 64 x 8 x 8

      # Third convolutional block
      nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output channels: 128
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),  # Output: 128 x 4 x 4

      # Flatten layer
      nn.Flatten(),

      # Fully connected layers
      nn.Linear(128 * 4 * 4, 256),  # Adjust input size based on previous layer's output
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(128, 3)  # Output layer: 3 classes
      )

  def forward(self, x):
    x = self.net(x)
    return x