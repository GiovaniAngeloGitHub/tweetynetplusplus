from pathlib import Path

from tweetynetplusplus.scripts.train_model import load_and_merge_configs

    
ROOT = Path(__file__).resolve().parents[3]
default_toml = ROOT / "configs" / "default.toml"
experiment_toml = ROOT / "configs" / "experiment_base.toml"
config = load_and_merge_configs(default_toml, experiment_toml)
