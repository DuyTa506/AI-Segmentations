from models.FCN import Custom_FCN
from models.Unet import UNet
#from models.PSPNet import PSPNet
from models.DeepLabv3 import DeepLabv3_plus
from models.Unet_Backbone import Unet_Backbone
from cityscapes import CityscapesLoader
from utils.augumentations import *
from metric.utils import AverageMeter, intersectionAndUnionGPU, runningScore
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.CE import cross_entropy2d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
net_h, net_w = 256, 512
batch_size = 5
n_epochs = 200
n_classes = 19

# Data augmentation
augment = Compose([RandomHorizontallyFlip(), RandomSized((0.625, 0.625)),
                   RandomRotate(15), RandomCrop((net_h, net_w))])

# Load datasets
local_path = "/app/duy55/segmentation/cityscapes"
train_data = CityscapesLoader(local_path, split="train", is_transform=True, augmentations=augment)
val_data = CityscapesLoader(local_path, split="val", is_transform=True, augmentations=None)

# train_data_size = len(train_data)
# val_data_size = len(val_data)
# # train_indices = np.random.choice(train_data_size, size=train_data_size // 2, replace=False)
# # val_indices = np.random.choice(val_data_size, size=val_data_size // 2, replace=False)

# train_subset = Subset(train_data, range(train_data_size // 2))
# val_subset = Subset(val_data, range(val_data_size // 2))

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=0)

model_name = "unet"
# Initialize model, loss function, and optimizer
if model_name == "unet" :
    model = UNet(n_classes=n_classes)
    #model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/model_best.pth"))
    model.to(device)
elif model_name == "fcn" : 
    model = Custom_FCN(n_classes=n_classes).to(device)
elif model_name == "re_unet" : 
    model = Unet_Backbone(n_classes=n_classes).to(device)
elif model_name == "PSP" : 
    model = PSPNet(layers=50, classes=n_classes).to(device)
elif model_name == "DeepLab" : 
    model = model = DeepLabv3_plus(n_classes=n_classes, pretrained = False, _print = True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

log_dir = "/app/duy55/segmentation/code/runs/Unet"
writer = SummaryWriter(log_dir=log_dir)


def train_and_validate(train_loader, val_loader, model, optimizer, epochs, writer):
    min_train_loss = float('inf') 
    
    for epoch_i in range(epochs):
        print(f"Epoch {epoch_i + 1}\n-------------------------------")
        
        # Training
        model.train()
        train_loss_meter = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred = model(images)
            loss = cross_entropy2d(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_meter.update(loss.item())
            
            if (i + 1) % 1 == 0:
                print(f'Train Epoch [{epoch_i + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}', end='\r')
        
        print(f'Train Epoch [{epoch_i + 1}/{epochs}], Average Loss: {train_loss_meter.avg:.4f}')
        
        writer.add_scalar('Train/Loss', train_loss_meter.avg, epoch_i)
        
        # Validation
        model.eval()
        print("Making evaluation :\n")
        val_score_meter = runningScore(n_classes)
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_pred = model(val_images)
                
                pred = val_pred.data.max(1)[1].cpu().numpy()
                gt = val_labels.data.cpu().numpy()
                val_score_meter.update(gt, pred)
        
        val_score_meter_val = val_score_meter.get_scores()
        print("Validation Metrics:")
        for key, value in val_score_meter_val.items():
            print(f'{key}: {value}')
        
        for key, value in val_score_meter_val.items():
            writer.add_scalar(f'Validation/{key}', value, epoch_i)
        
        if train_loss_meter.avg < min_train_loss:
            min_train_loss = train_loss_meter.avg 

            torch.save(model.state_dict(), f"{writer.log_dir}/model_best.pth")
            print("Saved model at epoch:", epoch_i + 1)
        
        # Reset meters for next epoch
        train_loss_meter.reset()
        val_score_meter.reset()


if __name__ == "__main__":
    train_and_validate(trainloader, valloader, model, optimizer, n_epochs, writer)
