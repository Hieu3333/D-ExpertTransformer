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

            # Flatten spatial dimensions for patch-level features
            B, C, H, W = feat1.shape
      
            patch_feats = torch.cat([feat1, feat2], dim=2)  # (B, 2048, 14,7)

            return patch_feats
        
        else:
            x = F.relu(self.features(x))  # Output shape: (B, 2048, 7, 7)
            return x
        
        



import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

class EfficientNet(nn.Module):
    def __init__(self, args):
        super(EfficientNet, self).__init__()
        self.dataset = args.dataset
        
        # Load EfficientNetV2-B0 without classifier
        self.model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
        
        # Remove global pooling and classification head
        self.model.global_pool = nn.Identity()  
        self.model.classifier = nn.Identity()
        
        # Define preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store reference to the second-to-last block
        self.num_blocks = len(self.model.blocks)
        self.second_to_last_block = self.model.blocks[-2]  # This refers to the block we want

    def forward(self, x):
        # Run through the initial layers
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        
        # Try different activation function names that might be used
        if hasattr(self.model, 'act1'):
            x = self.model.act1(x)
        elif hasattr(self.model, 'relu1'):
            x = self.model.relu1(x)
        else:
            # Fall back to a safer approach - you might need to check the actual name
            for name, module in self.model.named_children():
                if 'act' in name or 'relu' in name:
                    if name.endswith('1'):
                        x = module(x)
                        break
        
        # Run through all blocks up to and including the second-to-last block
        # The second-to-last block is at index -2
        for i in range(self.num_blocks - 1):  # Process all blocks except the last one
            x = self.model.blocks[i](x)
        print('shape:',x.shape)
        
        return x

class DenseNet(nn.Module):
    def __init__(self, args):
        super(DenseNet, self).__init__()
        self.dataset = args.dataset
        
        # Load DenseNet121
        densenet = models.densenet121(pretrained=True)
        
        # Extract features up to the final convolutional layer
        self.features = densenet.features  # (B, 1024, 7, 7)
        
        # Optional: freeze parameters
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