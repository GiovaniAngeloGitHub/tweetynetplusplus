import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from tweetynetplusplus.preprocessing.dataset_builder import BirdsongSpectrogramDataset
from tweetynetplusplus.training.metrics import compute_classification_metrics, plot_confusion_matrix
from tweetynetplusplus.models.backbones.resnet18_ft import ResNet18Finetune
from tweetynetplusplus.preprocessing.transforms import TemporalPadCrop, NormalizeTensor
from torchvision import transforms
from tweetynetplusplus.evaluation.reports import save_classification_report


def evaluate_from_checkpoint(
    model_path: str,
    processed_dir: str,
    annotation_file: str,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Transformações padrão
    transform = transforms.Compose([
        TemporalPadCrop(2048),
        NormalizeTensor()
    ])

    # Dataset completo (sem divisão treino/val/teste aqui)
    dataset = BirdsongSpectrogramDataset(processed_dir, annotation_file, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_names = [str(c) for c in dataset.le.classes_]

    # Modelo
    num_classes = len(dataset.le.classes_)
    model = ResNet18Finetune(num_classes=num_classes).to(device)
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

if __name__ == "__main__":
    evaluate_from_checkpoint(
        model_path="models_checkpoints/resnet18_20240523_141200.pt",
        processed_dir="data/processed/llb11",
        annotation_file="data/raw/llb11/llb11_annot.csv"
    )
