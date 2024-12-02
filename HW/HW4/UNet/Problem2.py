"""
Code written by Joey Wilson, 2023.
"""

import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

# Default transforms
def transform_train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def transform_test():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


# Data loader for the image segmentation data
class ImageSegmentation(Dataset):
    def __init__(self, root, split, transform=None, device="cpu"):
        self.root = root
        self.split = split
        self.transform = transform
        self.device = device

        self.dir = os.path.join(root, split)
        self.camera_files = sorted(os.listdir(os.path.join(self.dir, "Camera")))
        if self.split != "Test":
            self.seg_files = sorted(os.listdir(os.path.join(self.dir, "Labels")))

    def __len__(self):
        return len(self.camera_files)

    # Some good ideas here would be to crop a smaller section of the image
    # And add random flipping
    # Make sure the same augmentation is applied to image and label
    def image_augmentation(self, img_mat, label_mat):
        # Convert numpy arrays to PIL Images
        img = Image.fromarray(img_mat)
        label = Image.fromarray(label_mat)

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)

        # Random crop
        desired_height = 320
        desired_width = 1024
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(desired_height, desired_width))
        img = TF.crop(img, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Convert back to numpy arrays
        img_mat = np.array(img)
        label_mat = np.array(label)
        return img_mat, label_mat
    def image_augmentation(self, img_mat, label_mat):
        # Convert numpy arrays to PIL Images
        img = Image.fromarray(img_mat)
        label = Image.fromarray(label_mat)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)
        
        # Random rotation
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        label = TF.rotate(label, angle, interpolation=InterpolationMode.NEAREST)
        
        # Random crop
        desired_height = 320
        desired_width = 1024
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(desired_height, desired_width))
        img = TF.crop(img, i, j, h, w)
        label = TF.crop(label, i, j, h, w)
        
        # Color jitter (apply only to image)
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        img = color_jitter(img)
        
        # Convert back to numpy arrays
        img_mat = np.array(img)
        label_mat = np.array(label)
        return img_mat, label_mat

    # Return indexed item in dataset
    def __getitem__(self, index):
        file_name = os.path.join(self.dir, "Camera", self.camera_files[index])
        img = Image.open(file_name)
        img_mat = np.copy(np.asarray(img)[:, :, :3])
        if self.split != "Test":
            labeled_img = Image.open(os.path.join(self.dir, "Labels", self.seg_files[index]))
            label_mat = np.copy(np.asarray(labeled_img)[:, :, :3])
        else:
            label_mat = np.zeros_like(img_mat)
        if self.split == "Train":
            img_mat, label_mat = self.image_augmentation(img_mat, label_mat)
        return self.transform(img_mat), torch.tensor(label_mat, device=self.device)

    # Combine data within the batch
    def collate_fn(self, data):
        B = len(data)
        img_batch = torch.stack([data[i][0] for i in range(B)]).to(self.device)
        label_batch = torch.stack([data[i][1] for i in range(B)]).to(self.device)
        return img_batch, label_batch


# Basic convolution block with a 2D convolution, ReLU, and BatchNorm layer
# Conv with kernel size 2, stride 2, and padding 0 decreases the size of the image by half
# Conv with kernel size 3, stride 1, padding 1 keeps the image size constant
class ConvBlockStudent(nn.Module):
    def __init__(self, c_in, c_out, ds=False):
        super().__init__()
        layers = []
        if ds:
            layers.append(nn.Conv2d(c_in, c_out, 2, stride=2, padding=0))
        else:
            layers.append(nn.Conv2d(c_in, c_out, 3, stride=1, padding=1))
        layers.extend([
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# This is a basic U Net class. The decoder downsamples the image resolution at each level
# The encoder fuses information from the same resolution from the encoder at each level
# With a convolution operation. 

# In the encoder, we perform upsampling to ensure the same resolution with simple
# bilinear interpolation. An alternative to this is transposed convolution: 
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
class UNetStudent(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pre = ConvBlockStudent(3, 64)

        self.down1 = ConvBlockStudent(64, 128, ds=True)

        self.down2 = ConvBlockStudent(128, 256, ds=True)
        
        self.down3 = ConvBlockStudent(256, 512, ds=True)

        self.up2 = ConvBlockStudent(512+256, 256)
        
        self.up1 = ConvBlockStudent(256+128, 128)

        self.up0 = ConvBlockStudent(128+64, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x0 = self.pre(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Decoder
        x = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.up2(x)
        
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.up1(x)
        
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0, x], dim=1)
        x = self.up0(x)
        return self.out(x)


def IoU(targets, predictions, num_classes, ignore_index=0):
    # Initialize tensors
    intersections = torch.zeros(num_classes, device=targets.device)
    unions = torch.zeros_like(intersections)
    counts = torch.zeros_like(intersections)
    
    # Discard ignored points
    valid_mask = targets != ignore_index
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    
    # Loop over classes
    for c in range(num_classes):
        # Count occurrences of class c in targets
        counts[c] = (targets == c).sum()
        
        # Compute intersection and union
        intersection = ((predictions == c) & (targets == c)).sum()
        union = ((predictions == c) | (targets == c)).sum()
        
        # Store results
        intersections[c] = intersection
        unions[c] = union + 1e-5  # Add small value to avoid division by zero
    
    # Compute per-class IoU
    iou = intersections / unions
    
    # Set IoU to 1.0 for classes with no instances in targets
    iou[counts == 0] = 1.0
    
    # Exclude ignore_index from mean IoU calculation
    classes_to_include = torch.arange(num_classes, device=targets.device) != ignore_index
    miou = iou[classes_to_include].mean()
    
    return iou, miou