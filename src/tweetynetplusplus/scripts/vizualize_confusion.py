from tweetynetplusplus.training.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Exemplo de y_true e y_pred já salvos
    y_true = np.load("logs/y_true.npy")
    y_pred = np.load("logs/y_pred.npy")
    class_names = list(pd.read_csv("data/raw/llb11/llb11_annot.csv")["label"].unique())

    plot_confusion_matrix(y_true, y_pred, class_names, save_path="logs/confusion_matrix_manual.png")
