import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import timm
import torch.nn.functional as F
# Modify ResNet50 to stop before global avg pooling
class ResNet50(nn.Module):
    def __init__(self,args):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.dataset = args.dataset
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  # Keep until last conv layer
        # for param in resnet50.parameters():
        #     param.requires_grad = False
        
    def forward(self, x):
        if self.dataset == 'iu_xray':
            img1 = x[:, 0]  # (B, C, H, W)
            img2 = x[:, 1]

            # Extract features for both images
            feat1 = F.relu(self.features(img1))  # (B, 2048, 7, 7)
            feat2 = F.relu(self.features(img2))

            B, C, H, W = feat1.shape
      
            patch_feats = torch.cat([feat1, feat2], dim=2)  # (B, 2048, 14,7)

            return patch_feats
        
        else:
            x = F.relu(self.features(x))  # Output shape: (B, 2048, 7, 7)
            return x
        
        

class EfficientNet(nn.Module):
    def __init__(self,args):
        super(EfficientNet, self).__init__()
        self.dataset = args.dataset
        
        # Load EfficientNetV2-B0 without classifier
        self.model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
        self.model.global_pool = nn.Identity()  
        self.model.classifier = nn.Identity()   

        self.transform = transforms.Compose([
            transforms.Resize((356, 356)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.model(x)  # Returns shape (B, 1280, 12, 12)

class DenseNet(nn.Module):
    def __init__(self, args):
        super(DenseNet, self).__init__()
        self.dataset = args.dataset
        
    
        densenet = models.densenet121(pretrained=True)       
        self.features = densenet.features  # (B, 1024, 7, 7)
        

        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        if self.dataset == 'iu_xray':
            img1 = x[:, 0]  # (B, C, H, W)
            img2 = x[:, 1]

            feat1 = F.relu(self.features(img1))  # (B, 1024, 7, 7)
            feat2 = F.relu(self.features(img2))

            B, C, H, W = feat1.shape
            patch_feats = torch.cat([feat1, feat2], dim=2)  # (B, 1024, 14, 7)

            return patch_feats
        else:
            x = F.relu(self.features(x))  # (B, 1024, 7, 7)
            return x