import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# Modify ResNet50 to stop before global avg pooling
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  # Keep until last conv layer
        
        
    def forward(self, x):
        x = self.features(x)  # Output shape: (B, 2048, 7, 7)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)  # (B, 2048, 49)
        x = x.permute(0, 2, 1) # (B, 49, 2048) => N=49, H_I=2048
        return x

