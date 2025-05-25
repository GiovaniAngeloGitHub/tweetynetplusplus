from pathlib import Path
from tweetynetplusplus.evaluation.evaluate import evaluate_from_checkpoint
from tweetynetplusplus.scripts import config

if __name__ == "__main__":
    # Caminho do projeto base = dois níveis acima deste script
    ROOT_DIR = Path(__file__).resolve().parents[3]

    # Caminhos relativos
    model_path = ROOT_DIR / "models_checkpoints" / "resnet18_20250523_205523.pt"

    evaluate_from_checkpoint(
        model_path=model_path,
        config=config
    )
