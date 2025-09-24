import torchvision.models.video as video_models
import torch.nn as nn
from models.base_model import BaseModel

class ResNet3D(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = video_models.r3d_18(weights=None)  # no pretrained weights
        # Replace first conv to accept 2 channels instead of 3
        self.model.stem[0] = nn.Conv3d(
            in_channels=2,
            out_channels=64,
            kernel_size=(3,7,7),
            stride=(1,2,2),
            padding=(1,3,3),
            bias=False
        )
        # Replace classifier with sigmoid for binary classification
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
