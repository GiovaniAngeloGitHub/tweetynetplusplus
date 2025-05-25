from pathlib import Path
import tomli
from tweetynetplusplus.training.train import run_training_pipeline

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

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    default_toml = ROOT / "configs" / "default.toml"
    experiment_toml = ROOT / "configs" / "experiment_base.toml"
    config = load_and_merge_configs(default_toml, experiment_toml)

    run_training_pipeline(
        config=config
    )