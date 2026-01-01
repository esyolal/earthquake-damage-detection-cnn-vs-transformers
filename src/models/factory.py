from __future__ import annotations
import timm
import torch.nn as nn

def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    timm ile backbone + classification head.
    Binary için num_classes=2 tutuyoruz (softmax + cross entropy).
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
