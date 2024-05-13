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
import matplotlib.pyplot as plt
# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
n_classes = 19


# Load datasets
local_path = "/app/duy55/segmentation/cityscapes"
test_data = CityscapesLoader(local_path, split="test", is_transform=True, augmentations=None)


testloader = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=0)

model_name = "re_unet"
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



model_folder = f"/app/duy55/segmentation/code/demo_test/{model_name}"

os.makedirs(model_folder, exist_ok=True)
combined_samples  = []

num_samples = 5

saved_samples = 0

stop_flag = False

model.eval()
model.to(device)
with torch.no_grad():
    for image_num, (val_images,_) in tqdm(enumerate(testloader)):
        val_images = val_images.to(device)
        
        val_pred = model(val_images)
               
        prediction = val_pred.data.max(1)[1].cpu().numpy()

        for i in range(val_images.size(0)):
            decoded_pred = test_data.decode_segmap(prediction[i])
            
            combined_samples.append(decoded_pred)
            saved_samples += 1
            
            if saved_samples >= num_samples:
                stop_flag = True
                break
        
        if stop_flag:
            break

# Save the saved samples
for i, combined_image in enumerate(combined_samples):
    plt.imshow(combined_image)
    plt.axis('off')
    save_path = os.path.join(model_folder, f"sample_{i}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.clf()

print("Saved  samples successfully!")