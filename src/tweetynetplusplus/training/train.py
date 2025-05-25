import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from torchvision import transforms
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight
from tweetynetplusplus.preprocessing.dataset_builder import BirdsongSpectrogramDataset
from tweetynetplusplus.models.factory import get_model_from_config
from tweetynetplusplus.training.callbacks import EarlyStopping
from tweetynetplusplus.training.experiment_logger import init_logger, append_log
from tweetynetplusplus.training.metrics import compute_classification_metrics, plot_confusion_matrix
from tweetynetplusplus.preprocessing.transforms import TemporalPadCrop, NormalizeTensor
from tweetynetplusplus.config import settings


def save_metadata_training_model(final_model_path: str, train_metrics: dict, val_metrics: dict, num_classes: str, config: dict):

    meta_path = final_model_path.name.replace(".pt", ".meta.json")
    metadata = {
        "model_name": config["model"]["name"],
        "import_path": config["model"].get("import_path"),
        "pretrained": config["model"].get("pretrained", True),
        "num_classes": num_classes,
        "dataset": os.path.basename(config["data"]["processed_dir"]),
        "datetime": datetime.now().isoformat(),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📝 Metadados salvos em {meta_path}")


def run_training_pipeline(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        TemporalPadCrop(settings.data.target_width),
        NormalizeTensor()
    ])

    # Dataset & splits
    dataset = BirdsongSpectrogramDataset(
        config["data"]["processed_dir"],
        config["data"]["annotation_file"],
        transform=transform
    )
    total = len(dataset)
    train_len = int(0.7 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"])

    # Model
    num_classes = len(dataset.le.classes_)
    os.makedirs(config["logging"]["model_dir"], exist_ok=True)
    model_path = None
    model = get_model_from_config(
    model_name=config["model"]["name"],
    num_classes=num_classes,
    pretrained=config["model"].get("pretrained", True)
).to(device)

    if config["training"].get("use_saved_model", False):
        checkpoints = sorted([f for f in os.listdir(config["logging"]["model_dir"]) if f.endswith(".pt")])
        if checkpoints:
            model_path = os.path.join(config["logging"]["model_dir"], checkpoints[-1])
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.load_state_dict(torch.load(model_path, map_location=map_location))
            print(f"🔁 Loaded saved model: {model_path}")

    # Treinamento se necessário
    if not model_path:
        weights = compute_class_weight('balanced', classes=np.arange(num_classes)+1,
                                       y=[dataset.file_to_label[os.path.splitext(os.path.basename(f))[0]] for f in dataset.file_list])
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(),
                               lr=config["training"]["learning_rate"],
                               weight_decay=config["training"]["weight_decay"])

        early_stopper = EarlyStopping(patience=config["training"]["patience"])
        log_path = init_logger(config["logging"]["log_dir"])

        for epoch in range(config["training"]["epochs"]):
            model.train()
            total_loss, preds, targets = 0, [], []
            for batch in train_loader:
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
            print(f"[{epoch+1}] Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.3f} | F1: {train_metrics['f1']:.3f}")
            append_log(log_path, epoch + 1, "train", train_metrics)

            # Validação
            model.eval()
            val_loss, val_preds, val_targets = 0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    X = batch['spectrogram'].to(device)
                    y = batch['label'].to(device)
                    out = model(X)
                    val_loss += nn.CrossEntropyLoss()(out, y).item()
                    val_preds += out.argmax(1).tolist()
                    val_targets += y.tolist()

            val_metrics = compute_classification_metrics(val_targets, val_preds)
            val_metrics["loss"] = val_loss / len(val_loader)
            print(f"[{epoch+1}] Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.3f} | F1: {val_metrics['f1']:.3f}")
            append_log(log_path, epoch + 1, "val", val_metrics)

            if early_stopper(val_metrics['loss']):
                print("🛑 Early stopping ativado.")
                break

        # Salva o modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(config["logging"]["model_dir"], f"{config['model']['name']}_{timestamp}.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Modelo salvo em {final_model_path}")
        save_metadata_training_model(final_model_path, train_metrics, val_metrics, num_classes, config)
    # Avaliação final
    def eval_on_split(loader, name):
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
        print(f"\n📊 {name.upper()} | Acc: {metrics['acc']:.3f} | F1: {metrics['f1']:.3f} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
        if name == "teste":
            plot_confusion_matrix(targets, preds, list(dataset.le.classes_), save_path=os.path.join(config["logging"]["log_dir"], "confusion_matrix.png"))

    eval_on_split(val_loader, "validacao")
    eval_on_split(test_loader, "teste")
