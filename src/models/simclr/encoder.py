from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TimmEncoder(nn.Module):
    """
    timm backbone feature extractor.
    Çıkış her zaman (B, C).

    Handles:
    - (B, C) -> 그대로
    - (B, N, C) -> mean over N
    - (B, C, H, W) -> GAP over H,W
    - (B, H, W, C) -> mean over H,W
    """
    def __init__(self, model_name: str, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # feature dim'i sağlam bul: dummy forward
        self.feature_dim = self._infer_feature_dim(img_size)

    def _to_bc(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 2:
            # (B, C)
            return feats
        if feats.dim() == 3:
            # (B, N, C)
            return feats.mean(dim=1)
        if feats.dim() == 4:
            # (B, C, H, W) OR (B, H, W, C)
            # ikinci boyut küçükse (C genelde 96/192/384/768), H/W 7/14/28 vs
            b, d1, d2, d3 = feats.shape

            # Heuristic: channel dim genelde en büyük olanlardan biri olur.
            # Eğer son dim "C" gibi duruyorsa: (B,H,W,C)
            if d3 >= d1 and d3 >= d2:
                return feats.mean(dim=(1, 2))  # mean over H,W

            # Eğer ikinci dim "C" gibi duruyorsa: (B,C,H,W)
            return feats.mean(dim=(2, 3))  # GAP over H,W

        raise ValueError(f"Unexpected feature shape: {tuple(feats.shape)}")

    @torch.no_grad()
    def _infer_feature_dim(self, img_size: int) -> int:
        device = next(self.model.parameters()).device
        x = torch.zeros(2, 3, img_size, img_size, device=device)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        feats = self._to_bc(feats)
        return int(feats.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)
        feats = self._to_bc(feats)
        return feats
