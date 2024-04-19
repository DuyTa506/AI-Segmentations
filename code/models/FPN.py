import torch
import torch.nn as nn
import torchvision
import timm

class myFPNSegmentation(nn.Module):
    def __init__(self, backbone_name, fpn_channel1, fpn_channel2, n_classes):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.n_classes = n_classes
        self.fpn_channel1 = fpn_channel1
        self.fpn_channel2 = fpn_channel2
        
        # Get feature channels from the backbone model dynamically
        self.in_fpn_channel_lst = [info["num_chs"] for info in self.backbone.feature_info]
        
        self.fpn1 = torchvision.ops.FeaturePyramidNetwork(self.in_fpn_channel_lst, self.fpn_channel1)
        self.fpn2 = torchvision.ops.FeaturePyramidNetwork([self.fpn_channel1]*len(self.in_fpn_channel_lst), self.fpn_channel2)
        self.conv_cls = nn.Conv2d(self.fpn_channel2, self.n_classes, 1, 1, 0)

    def forward(self, x):
        ori_size = x.shape[2:] # (H, W)
        features = {}
        backbone_features = self.backbone(x)
        for i, feature_map in enumerate(backbone_features):
            features[f"x_{i+1}"] = feature_map
        
        fpn_features1 = self.fpn1(features)
        fpn_features2 = self.fpn2(fpn_features1)
        
        x = fpn_features2["0"]
        for i in range(1, len(fpn_features2)):
            x += torch.nn.functional.interpolate(fpn_features2[str(i)], size=fpn_features2["0"].shape[2:], mode="bilinear")
        x = torch.nn.functional.interpolate(x, size=ori_size, mode="bilinear")
        x = self.conv_cls(x)
        return x

if __name__ == '__main__':
    model = myFPNSegmentation("resnet50d", 256, 128, 19)
    model.eval()
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
    print(y.shape)
