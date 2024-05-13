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
n_classes = 19


# Load datasets
local_path = "/app/duy55/segmentation/cityscapes"
test_data = CityscapesLoader(local_path, split="val", is_transform=True, augmentations=None)


testloader = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=0)

model_name = "re_unet"
# Initialize model, loss function, and optimizer
if model_name == "unet" :
    model = UNet(n_classes=n_classes)
    model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/Unet/model_best.pth"))
    model.to(device)
elif model_name == "fcn" : 
    model = Custom_FCN(n_classes=n_classes)
    model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/FCN/model_best.pth"))
    model.to(device)
elif model_name == "re_unet" : 
    model = Unet_Backbone(n_classes=n_classes)
    model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/ResUnet/model_best.pth"))
    model.to(device)
elif model_name == "PSP" : 
    model = PSPNet(layers=50, classes=n_classes)
    model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/PSP/model_best.pth"))
    model.to(device)
elif model_name == "DeepLab" : 
    model = DeepLabv3_plus(n_classes=n_classes, pretrained = False, _print = True)
    model.load_state_dict(torch.load("/app/duy55/segmentation/code/runs/DeepLab/model_best.pth"))
    model.to(device)

print("Model loaded sucessfully !")
log_dir = "/app/duy55/segmentation/code/runs/val/{}".format(model_name)
writer = SummaryWriter(log_dir=log_dir)


def test(test_loader, model, device, writer=None):
    model.eval()
    test_score_meter = runningScore(n_classes)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model(images)
            pred = pred.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            
            test_score_meter.update(gt, pred)
    
    test_score = test_score_meter.get_scores()
    
    print("Test Metrics:")
    for key, value in test_score.items():
        print(f'{key}: {value}')
    
    if writer is not None:
        for key, value in test_score.items():
            writer.add_scalar(f'Test/{key}', value)
    
    return test_score


if __name__ == "__main__":
    test_score = test(testloader, model, device, writer)
