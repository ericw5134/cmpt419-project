import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VideoCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoCNN, self).__init__()
        # Use a pretrained ResNet and modify for video
        resnet = models.resnet18(pretrained=True)

        # Modify first conv to accept grayscale
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fc layer
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        # Temporal pooling and classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [batch, num_frames, C, H, W]
        batch_size, num_frames, C, H, W = x.shape

        # Merge batch and time to extract frame-wise features
        x = x.view(batch_size * num_frames, C, H, W)  # [B*N, C, H, W]
        features = self.feature_extractor(x)  # [B*N, 512, 1, 1]
        features = features.view(batch_size, num_frames, -1)  # [B, N, 512]

        # Temporal pooling (mean over frames)
        features = features.mean(dim=1)  # [B, 512]

        return self.classifier(features)  # [B, num_classes]


class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.LazyLinear(128)  # automatically infers input features on first forward
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)