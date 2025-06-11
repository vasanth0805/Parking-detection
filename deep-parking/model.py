import torch
import torch.nn as nn
import torchvision.models as models

class ParkingNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ParkingNet, self).__init__()
        # Use a pre-trained AlexNet as base
        self.alexnet = models.alexnet(pretrained=True)
        # Modify the classifier for our binary classification task
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)
