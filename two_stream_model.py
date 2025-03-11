import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStreamCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(SpatialStreamCNN, self).__init__()
        # Use a pretrained ResNet as the backbone
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjust the final layer

    def forward(self, x):
        return self.resnet(x)

class TemporalStreamCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(TemporalStreamCNN, self).__init__()
        # Use a simple 3D convolutional model or other temporal processing models
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(64 * 112 * 112 * 8, num_classes)  # Adjust based on your input size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
