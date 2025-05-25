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
from tweetynetplusplus.config import settings

def evaluate_from_checkpoint(
    model_path: str,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    config: dict = None
):
    # Transformações padrão
    transform = transforms.Compose([
        TemporalPadCrop(settings.data.target_width),
        NormalizeTensor()
    ])

    # Dataset completo (sem divisão treino/val/teste aqui)
    dataset = BirdsongSpectrogramDataset(config["data"]["processed_dir"], config["data"]["annotation_file"], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_names = [str(c) for c in dataset.le.classes_]

    # Modelo
    num_classes = len(dataset.le.classes_)
    model = get_model_from_config(
        model_name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"].get("pretrained", True)
    ).to(device)
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()

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

    # Matriz de confusão
    plot_confusion_matrix(all_targets, all_preds, class_names, save_path="logs/eval_confusion_matrix.png")
    print("\nMatriz de confusão salva em logs/eval_confusion_matrix.png")

    # Relatório em CSV e JSON
    save_classification_report(all_targets, all_preds, class_names)
    np.save("logs/y_true.npy", np.array(all_targets))
    np.save("logs/y_pred.npy", np.array(all_preds))

