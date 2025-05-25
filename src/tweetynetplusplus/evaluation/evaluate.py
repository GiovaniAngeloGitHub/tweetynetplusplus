import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from tweetynetplusplus.preprocessing.dataset_builder import BirdsongSpectrogramDataset
from tweetynetplusplus.training.metrics import compute_classification_metrics, plot_confusion_matrix
from tweetynetplusplus.models.factory import get_model_from_config
from tweetynetplusplus.preprocessing.transforms import TemporalPadCrop, NormalizeTensor
from torchvision import transforms
from tweetynetplusplus.evaluation.reports import save_classification_report

import json


def evaluate_from_checkpoint(
    model_path: str,
    processed_dir: str,
    annotation_file: str,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    config: dict = None
):
    transform = transforms.Compose([
        TemporalPadCrop(2048),
        NormalizeTensor()
    ])

    dataset = BirdsongSpectrogramDataset(processed_dir, annotation_file, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_names = [str(c) for c in dataset.le.classes_]

    num_classes = len(dataset.le.classes_)
    model = get_model_from_config(
        model_name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True),
        import_path=config["model"].get("import_path", None)
    ).to(device)

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()

    # 🔍 Se houver .meta.json correspondente, exibe as informações
    meta_path = model_path.name.replace(".pt", ".meta.json")
    if os.path.exists(meta_path):
        print("\n📄 Carregando metadados do experimento:")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        print(json.dumps(metadata, indent=2))

    # Avaliação
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch['spectrogram'].to(device)
            y = batch['label'].to(device)
            out = model(X)
            all_preds += out.argmax(1).tolist()
            all_targets += y.tolist()

    metrics = compute_classification_metrics(all_targets, all_preds)
    print("\n=== Avaliação ===")
    print(f"Accuracy: {metrics['acc']:.3f}")
    print(f"F1 Score (macro): {metrics['f1']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")

    print("\n=== Classification Report ===")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

    plot_confusion_matrix(all_targets, all_preds, class_names, save_path="logs/eval_confusion_matrix.png")
    print("\nMatriz de confusão salva em logs/eval_confusion_matrix.png")

    save_classification_report(all_targets, all_preds, class_names)
    np.save("logs/y_true.npy", np.array(all_targets))
    np.save("logs/y_pred.npy", np.array(all_preds))