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
            feat1 = F.relu(self.features(img1), inplace=True)  # (B, 2048, 7, 7)
            feat2 = F.relu(self.features(img2), inplace=True)

            # Flatten spatial dimensions for patch-level features
            B, C, H, W = feat1.shape
            feat1 = feat1.view(B, C, -1).permute(0, 2, 1)  # (B, 49, 2048)
            feat2 = feat2.view(B, C, -1).permute(0, 2, 1)

            patch_feats = torch.cat([feat1, feat2], dim=1)  # (B, 98, 2048)

            return patch_feats
        
        else:
            x = F.relu(self.features(x))  # Output shape: (B, 2048, 7, 7)
            return x
        
        



class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        
        # Load EfficientNetV2-B0 without classifier
        self.model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
        self.model.global_pool = nn.Identity()  # Remove global pooling
        self.model.classifier = nn.Identity()   # Remove classification head

        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((356, 356)),  # Resize to match input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.model(x)  # Returns feature map of shape (B, 1280, 12, 12)

    