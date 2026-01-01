from __future__ import annotations
import torch
import torch.nn.functional as F

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss.
    AMP/fp16 ile uyumlu olacak şekilde similarity hesaplarını float32 yapıyoruz.
    z1, z2: (B, D)
    """
    B = z1.size(0)

    # normalize (fp16 olabilir, sorun değil)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # concat
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # similarity'i float32 hesapla (AMP overflow önleme)
    sim = (z.float() @ z.float().T) / float(temperature)  # (2B, 2B)

    # diagonal mask
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)  # fp16 için güvenli negatif

    # positive indices: i <-> i+B
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)

    # cross_entropy float32 üzerinde stabil
    loss = F.cross_entropy(sim, pos)
    return loss
