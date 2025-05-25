import os
import csv
from typing import Dict

def init_logger(log_dir: str, filename: str = "training_log.csv") -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "split", "loss", "accuracy", "f1"])
    return path

def append_log(log_path: str, epoch: int, split: str, metrics: Dict[str, float]):
    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            epoch,
            split,
            f"{metrics['loss']:.4f}",
            f"{metrics['acc']:.4f}",
            f"{metrics['f1']:.4f}",
        ])
