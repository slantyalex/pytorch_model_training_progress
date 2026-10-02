import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AnimalVisionModel(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)