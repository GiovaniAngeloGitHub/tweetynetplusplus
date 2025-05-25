from pathlib import Path
from tweetynetplusplus.evaluation.evaluate import evaluate_from_checkpoint

if __name__ == "__main__":
    # Caminho do projeto base = dois níveis acima deste script
    ROOT_DIR = Path(__file__).resolve().parents[3]

    # Caminhos relativos
    model_path = ROOT_DIR / "models_checkpoints" / "resnet18_20250523_205523.pt"
    processed_dir = ROOT_DIR / "data" / "processed" / "llb11"
    annotation_file = ROOT_DIR / "data" / "raw" / "llb11" / "llb11_annot.csv"

    evaluate_from_checkpoint(
        model_path=str(model_path),
        processed_dir=str(processed_dir),
        annotation_file=str(annotation_file)
    )
