from pathlib import Path
from tweetynetplusplus.training.train import run_training_pipeline
from tweetynetplusplus.utils.func import load_and_merge_configs


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    default_toml = ROOT / "configs" / "default.toml"
    experiment_toml = ROOT / "configs" / "experiment_base.toml"
    config = load_and_merge_configs(default_toml, experiment_toml)

    run_training_pipeline(
        config=config
    )