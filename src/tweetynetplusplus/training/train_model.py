import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from tweetynetplusplus.preprocessing.dataset_builder import BirdsongSpectrogramDataset
from tweetynetplusplus.models.backbones.resnet18_ft import ResNet18Finetune
from tweetynetplusplus.training.callbacks import EarlyStopping
from tweetynetplusplus.training.experiment_logger import init_logger, append_log
from tweetynetplusplus.training.metrics import compute_classification_metrics, plot_confusion_matrix
from tweetynetplusplus.preprocessing.transforms import TemporalPadCrop, NormalizeTensor
from torchvision import transforms

def prepare_datasets(processed_dir, annotation_file, transform, split=(0.7, 0.15, 0.15)):
    dataset = BirdsongSpectrogramDataset(processed_dir, annotation_file, transform=transform)
    total = len(dataset)
    train_len = int(split[0] * total)
    val_len = int(split[1] * total)
    test_len = total - train_len - val_len
    return dataset, random_split(dataset, [train_len, val_len, test_len])

def create_dataloaders(train_ds, val_ds, test_ds, batch_size):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )

def build_model(num_classes, model_dir, use_saved_model, device):
    os.makedirs(model_dir, exist_ok=True)
    if use_saved_model:
        checkpoints = sorted([f for f in os.listdir(model_dir) if f.endswith(".pt")])
        if checkpoints:
            model_path = os.path.join(model_dir, checkpoints[-1])
            print(f"🔁 Carregando modelo salvo: {model_path}")
            model = ResNet18Finetune(num_classes=num_classes).to(device)
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=map_location))
            return model, model_path
        print("⚠️ Nenhum modelo salvo encontrado. Treinando do zero.")
    model = ResNet18Finetune(num_classes=num_classes).to(device)
    return model, None

def compute_class_weights(dataset, num_classes, device):
    class_labels = [
        dataset.file_to_label[os.path.splitext(os.path.basename(f))[0]]
        for f in dataset.file_list
    ]
    weights = compute_class_weight('balanced', classes=np.arange(num_classes)+1, y=class_labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, model_dir, log_path):
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        total_loss, preds, targets = 0, [], []

        for batch in tqdm(train_loader, desc=f"[{epoch+1}/{num_epochs}] Treinando"):
            X = batch['spectrogram'].to(device)
            y = batch['label'].to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds += out.argmax(1).tolist()
            targets += y.tolist()

        train_metrics = compute_classification_metrics(targets, preds)
        train_metrics["loss"] = total_loss / len(train_loader)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.3f} | F1: {train_metrics['f1']:.3f}")
        append_log(log_path, epoch+1, "train", train_metrics)

        model.eval()
        val_loss, val_preds, val_targets = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['spectrogram'].to(device)
                y = batch['label'].to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_preds += out.argmax(1).tolist()
                val_targets += y.tolist()

        val_metrics = compute_classification_metrics(val_targets, val_preds)
        val_metrics["loss"] = val_loss / len(val_loader)
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.3f} | F1: {val_metrics['f1']:.3f}")
        append_log(log_path, epoch+1, "val", val_metrics)

        if early_stopping(val_metrics['loss']):
            print("🛑 Early stopping ativado.")
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(model_dir, f"resnet18_{timestamp}.pt")
    torch.save(model.state_dict(), path)
    print(f"✅ Modelo salvo em {path}")
    return model

def evaluate_model(model, loader, device, split_name, class_names):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch['spectrogram'].to(device)
            y = batch['label'].to(device)
            out = model(X)
            preds += out.argmax(1).tolist()
            targets += y.tolist()

    metrics = compute_classification_metrics(targets, preds)
    print(f"\n📊 {split_name} | Acc: {metrics['acc']:.3f} | F1: {metrics['f1']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")

    if split_name == "TESTE":
        plot_confusion_matrix(targets, preds, class_names, save_path="logs/confusion_matrix.png")


def train_model(
    processed_dir: str,
    annotation_file: str,
    model_dir: str = "models_checkpoints/",
    num_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    use_saved_model: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    transform = transforms.Compose([
        TemporalPadCrop(2048),
        NormalizeTensor()
    ])

    dataset, (train_ds, val_ds, test_ds) = prepare_datasets(processed_dir, annotation_file, transform)
    num_classes = len(dataset.le.classes_)
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, batch_size)

    model, loaded_path = build_model(num_classes, model_dir, use_saved_model, device)
    log_path = init_logger("logs")

    if not use_saved_model or not loaded_path:
        weights_tensor = compute_class_weights(dataset, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model = train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, model_dir, log_path)

    evaluate_model(model, val_loader, device, "VALIDACAO", class_names=dataset.le.classes_)
    evaluate_model(model, test_loader, device, "TESTE", class_names=dataset.le.classes_)
