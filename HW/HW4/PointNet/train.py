import os
import sys
import time
import numpy as np
import torch
import yaml
from tqdm import tqdm
from utils import *
from Problem1 import *
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim



GOOGLE_DRIVE_PATH = os.path.join('/', 'home', 'fishros', 'ROB535', 'HW', 'HW4', 'PointNet')
sys.path.append(GOOGLE_DRIVE_PATH)
os.environ["TZ"] = "US/Eastern"
time.tzset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(GOOGLE_DRIVE_PATH)
config_file = os.path.join(DATA_PATH, "semantic_kitti.yaml")
kitti_config = yaml.safe_load(open(config_file, 'r'))
# Label map
LABELS_REMAP = kitti_config["learning_map"]
# File paths
demo_pc = os.path.join(DATA_PATH, "Train", "velodyne_ds", "000000.bin")
demo_label = os.path.join(DATA_PATH, "Train", "labels_ds", "000000.label")
# Obtain numpy arrays
demo_pc = np.fromfile(demo_pc, dtype=np.float32).reshape(-1, 4)
demo_label = np.fromfile(demo_label, dtype=np.int32).reshape(-1) & 0xFFFF
# Remap labels
label_remap = get_remap_lut(LABELS_REMAP)
demo_label = label_remap[demo_label]

# Hyperparameters: try changing these
lr = 0.0001
num_epochs = 100
batch_size = 4

# Channel sizes: try changing these
cs_t_en = [3, 64, 128, 1024]
cs_t_dec = [1024, 512, 256]
cs_enc = [3, 128, 256, 512, 1024]
cs_dec = [1024 + 1024, 512, 256]

# Data loaders
trainset = PointLoader(os.path.join(DATA_PATH, "Train"), label_remap,
                       device=device, data_split="Train")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=trainset.collate_fn)

valset = PointLoader(os.path.join(DATA_PATH, "Val"), label_remap,
                     device=device, data_split="Val")
val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                          shuffle=True,
                                          collate_fn=valset.collate_fn)

testset = PointLoader(os.path.join(DATA_PATH, "Test"), label_remap,
                     device=device, data_split="Test")
test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False,
                                          collate_fn=testset.collate_fn)

def train_net_iou(net, trainloader, val_loader, device, num_epochs, optimizer, criterion):
  for epoch in range(num_epochs):  # loop over the dataset multiple times
      # Train
      net.train()
      total_loss = 0
      i = 0
      loop = tqdm(trainloader)
      for data in loop:
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = net(inputs)
          B, N, C = outputs.shape
          outputs = outputs.view(-1, C)
          labels = labels.view(-1).long()


          # backward + optimize
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          loop.set_description("Training")
          total_loss += loss.item()
          i += 1
          loop.set_postfix(loss=total_loss / i)

      # Validate
      all_targets = []
      all_preds = []
      net.eval()
      loop = tqdm(val_loader)
      loop.set_description("Validation")
      with torch.no_grad():
        for data in loop:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Forward pass
            outputs = net(inputs)
            B, N, C = outputs.shape
            outputs = outputs.view(-1, C)
            labels = labels.view(-1).long()

            # Targets and predictions for iou
            _, predicted = torch.max(outputs, 1)
            all_targets.append(labels)
            all_preds.append(predicted)
      iou, miou = IoU(torch.concatenate(all_targets), torch.concatenate(all_preds), 20)

      # print statistics
      print(f'epochs: {epoch + 1} mIoU Val: {100 * miou.item():.3f}')
  print('Finished Training')
  
  # Training will be very slow on CPU, recommend using GPU
seed_torch()


net = PointNetFull(cs_enc, cs_dec, cs_t_en, cs_t_dec).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

train_net_iou(net, trainloader, val_loader, device, num_epochs, optimizer, criterion)

# Save Weights
PATH = os.path.join(GOOGLE_DRIVE_PATH, 'YourNet.pth')
torch.save(net.state_dict(), PATH)

# Generate Predictions
net.load_state_dict(torch.load(PATH))

# Test
i = 0
net.eval()
save_dir = os.path.join(DATA_PATH, "Test", "Problem1_Predictions")
if not os.path.exists(save_dir):
  os.mkdir(save_dir)
with torch.no_grad():
  for inputs, __ in iter(testset):
    # Get the inputs; data is a list of [inputs, labels]
    input = torch.from_numpy(inputs).to(device)
    input = torch.unsqueeze(input, 0)
    
    # Forward
    output = net(input).squeeze(0)
    _, predicted = torch.max(output, 1)
    
    # Save predictions
    if i % 10 == 0:
      predictions_np = predicted.detach().cpu().numpy().astype(np.int32)
      save_path = os.path.join(save_dir, str(i).zfill(6) + ".label")
      predictions_np.tofile(save_path)
    i += 1