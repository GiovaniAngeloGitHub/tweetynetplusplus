from pathlib import Path
import tomli
from tweetynetplusplus.evaluation.evaluate import evaluate_from_checkpoint

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[3]

    # Carrega config
    config_path = ROOT_DIR / "configs" / "default.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    # Procura o último modelo salvo
    model_dir = ROOT_DIR / config["logging"]["model_dir"]
    checkpoints = sorted([ckpt for ckpt in model_dir.glob("*.pt")])
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