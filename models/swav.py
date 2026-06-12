import torch
import torch.nn as nn
import torch.nn.functional as F

class SwAV(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        backbone.fc = nn.Identity()
        self.backbone = backbone
    
    def forward(self, x1, x2):
        # SwAV is not implemented in this repo yet.
        raise NotImplementedError("SwAV is not implemented yet.")





