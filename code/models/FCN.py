import torch
import torch.nn as nn

def down_conv(small_channels, big_channels, pad):   ### contracting block
    return torch.nn.Sequential(
        torch.nn.Conv2d(small_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels),
        torch.nn.Conv2d(big_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels)
    )   ## consider stride = 2

def up_conv(big_channels, small_channels, pad):
    return torch.nn.Sequential(
        torch.nn.Conv2d(big_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels),
        torch.nn.Conv2d(small_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels)
    )


class Custom_FCN(nn.Module):
    def crop(self, a, b):
        ## a, b tensor shape = [batch, channel, H, W]
        Ha = a.size()[2]
        Wa = a.size()[3]
        Hb = b.size()[2]
        Wb = b.size()[3]

        adapt = torch.nn.AdaptiveMaxPool2d((Ha,Wa))
        crop_b = adapt(b) 
            
        return crop_b    
   
    
    def __init__(self, n_classes):
        super().__init__()

        self.relu    = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)         
        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.std = torch.Tensor([0.25, 0.25, 0.25])
        
        a = 32
        b = a*2 #64
        c = b*2 #128
        d = c*2 #256
        
        n_class = n_classes
        
        self.conv_down1 = down_conv(3, a, 1) # 3 --> 32
        self.conv_down2 = down_conv(a, b, 1)  # 32 --> 64
        self.conv_down3 = down_conv(b, c, 1)  # 64 --> 128
        self.conv_down4 = down_conv(c, d, 1)  # 128 --> 256
        
        self.bottleneck = torch.nn.ConvTranspose2d(d, c, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.conv_up3 = up_conv(c, b, 1)  # 128 --> 64
        self.upsample3 = torch.nn.ConvTranspose2d(b, a, kernel_size=3, stride=2, padding=1, output_padding=1)   
                 
        self.classifier = torch.nn.Conv2d(a, n_class, kernel_size=1) 
        
    
    def forward(self, x):
        H = x.shape[2]
        W = x.shape[3]
        z = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)
        #################### DOWN / ENCODER #############################
        conv1 =  self.conv_down1(z)   # 3 --> 32
        mx1 = self.maxpool(conv1)
        conv2 =  self.conv_down2(mx1)  # 32 --> 64
        mx2 = self.maxpool(conv2) 
        conv3 =  self.conv_down3(mx2) # 64 --> 128  
        mx3 = self.maxpool(conv3) 
        conv4 =  self.conv_down4(conv3) # 128 --> 256  ################### CHANGED THIS

        ########################### BOTTLENECK #############################
        score = self.bottleneck(conv4)  # 256 --> 128
       
        ######################### UP/DECODER #######################
        crop_conv3 = self.crop(score, conv3)    
        score = score + crop_conv3   ### add 128 
        
        ##########################
        score = self.conv_up3(score)  # 128 --> 64
        score = self.upsample3(score)  # 64 --> 32     
        crop_conv1 = self.crop(score, conv1)   
        score = score + crop_conv1   ### add 32           
        
        ############################
        score = self.classifier(score) 
        out = torch.nn.functional.interpolate(score, size=(H,W))
        out = out[:, :, :H, :W]
        return out

if __name__ == '__main__':
    model = Custom_FCN(n_classes=19)
    model.eval()
    x = torch.rand(2, 3, 448, 448)
    y = model(x)
    print(y.shape)
        
  