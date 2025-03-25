import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
from model_flashinterimage import fusion_net

from RetinexFormer_arch import RetinexFormer


class final_net(nn.Module):
    def __init__(self):
        super(final_net, self).__init__()
        self.dehazing_model = fusion_net()
    def forward(self, hazy, scale=0.05):

        x = self.dehazing_model(hazy)
    
        return x + hazy
