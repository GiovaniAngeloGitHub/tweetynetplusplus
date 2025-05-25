import torch
import torch.nn.functional as F

def resize_tensor(t: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Redimensiona tensor (1, H, W) para (1, target_bins, W) via interpolação bilinear."""
    return F.interpolate(t.unsqueeze(0), size=(target_bins, t.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)