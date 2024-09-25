import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import src.utils.imports as im

im.ssl._create_default_https_context = im.ssl._create_unverified_context

class VisualTransformerModel(nn.Module):
    def __init__(self, num_classes=1):
        super(VisualTransformerModel, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        
        num_ftrs = self.vit.heads[-1].in_features
        
        self.vit.heads[-1] = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.vit(x)

def build_vit_model(num_classes=1):
    """
    Builds a Visual Transformer model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        VisualTransformerModel: PyTorch model of Visual Transformer network.
    """
    return VisualTransformerModel(num_classes)