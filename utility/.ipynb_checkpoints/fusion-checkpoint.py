import torch
import torch.nn as nn

# A simple late fusion model that combines the predictions of two models
class LateFusion(nn.Module):
    def __init__(self, num_classes=5, fusion_type='mlp'):
        super(LateFusion, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * 2, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes)
            )
        elif fusion_type == 'average':
            self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, video_pred, audio_pred):
        if self.fusion_type == 'average':
            return (video_pred + audio_pred) / 2
        else:
            combined = torch.cat((video_pred, audio_pred), dim=1)
            return self.fusion(combined)