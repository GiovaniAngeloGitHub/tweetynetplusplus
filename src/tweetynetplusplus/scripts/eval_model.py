from pathlib import Path
import tomli
from tweetynetplusplus.evaluation.evaluate import evaluate_from_checkpoint
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[3]
    # Carrega config
    config_path = ROOT_DIR / "configs" / "default_efficientnet.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    logger.info("📋 Configuração carregada:")
    logger.info(json.dumps(config, indent=4))

    # Procura o último modelo salvo
    logger.info("🔍 Procurando o ultimo modelo salvo em models_checkpoints/")
    logger.info(f"📂 {config['logging']['model_dir']}")
    model_dir = ROOT_DIR / config["logging"]["model_dir"]

    checkpoints = sorted([ckpt for ckpt in model_dir.glob(f"{config['model']['name']}*.pt")])
    if not checkpoints:
        raise FileNotFoundError("Nenhum modelo .pt encontrado em models_checkpoints/")
    model_path = checkpoints[-1]


    # Avaliação
    evaluate_from_checkpoint(
        model_path=model_path,
        processed_dir=config["data"]["processed_dir"],
        annotation_file=config["data"]["annotation_file"],
        batch_size=config["data"]["batch_size"],
        config=config
    )