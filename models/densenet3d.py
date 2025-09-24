import torch.nn as nn
from torchvision.models.video import mc3_18
from .base_model import BaseModel

class DenseNet3D(BaseModel):
    """
    Wrapper for 3D DenseNet-like model using torchvision's mc3_18.
    Adapted for binary classification (ASD vs Control).
    """

    def __init__(self):
        super(DenseNet3D, self).__init__()
        self.model = mc3_18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
