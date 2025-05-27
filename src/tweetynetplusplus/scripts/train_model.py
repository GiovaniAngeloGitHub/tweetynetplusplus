import json
import logging
from pathlib import Path
from tweetynetplusplus.training.train import run_training_pipeline
from tweetynetplusplus.utils.func import load_and_merge_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    default_toml = ROOT / "configs" / "default_resnet50.toml"
    experiment_toml = ROOT / "configs" / "experiment_base.toml"
    config = load_and_merge_configs(default_toml, experiment_toml)

    logger.info("📋 Configuração carregada:")
    logger.info(json.dumps(config, indent=4))

    run_training_pipeline(
        config=config
    )