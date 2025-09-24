import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from .base_model import BaseModel

class R2Plus1D(BaseModel):
    """
    Wrapper for 3D R(2+1)D model using torchvision's r2plus1d_18.
    Adapted for binary classification.
    """

    def __init__(self):
        super(R2Plus1D, self).__init__()
        self.model = r2plus1d_18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
