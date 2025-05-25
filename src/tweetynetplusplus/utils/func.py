import tomli
import torch
import torch.nn.functional as F

def resize_tensor(t: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Redimensiona tensor (1, H, W) para (1, target_bins, W) via interpolação bilinear."""
    return F.interpolate(t.unsqueeze(0), size=(target_bins, t.shape[-1]), mode='bilinear', align_corners=False).squeeze(0)

def load_and_merge_configs(default_path: str, experiment_path: str) -> dict:
    with open(default_path, "rb") as f:
        default_cfg = tomli.load(f)
    with open(experiment_path, "rb") as f:
        exp_cfg = tomli.load(f)

    def merge_dicts(d1, d2):
        for k, v in d2.items():
            if isinstance(v, dict) and k in d1:
                merge_dicts(d1[k], v)
            else:
                d1[k] = v
        return d1

    return merge_dicts(default_cfg, exp_cfg)