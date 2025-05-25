import os
import numpy as np
from pathlib import Path
import pandas as pd
from tweetynetplusplus.evaluation.reports import save_classification_report

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]
    logs_dir = ROOT / "logs"

    y_true_path = logs_dir / "y_true.npy"
    y_pred_path = logs_dir / "y_pred.npy"

    if not y_true_path.exists() or not y_pred_path.exists():
        raise FileNotFoundError("❌ Arquivos y_true.npy e y_pred.npy não encontrados. Rode evaluate_model.py antes.")

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    class_names = list(pd.read_csv(ROOT / "data/raw/llb11/llb11_annot.csv")["label"].unique())

    save_classification_report(y_true, y_pred, class_names, output_dir=str(logs_dir))
