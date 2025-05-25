import os
import torch
import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any
from glob import glob
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class BirdsongSpectrogramDataset(Dataset):
    """
    Dataset PyTorch para espectrogramas de canto de pássaros (.pt) com anotações CSV.

    Args:
        processed_dir (str): Caminho com arquivos .pt (espectrogramas)
        annotation_file (str): CSV com colunas 'audio_file' e 'label'
        transform (callable, optional): Transformações a aplicar nos espectrogramas
    """
    def __init__(self, processed_dir: str, annotation_file: str, transform: Optional[Callable] = None):
        super().__init__()
        self.processed_dir = processed_dir
        self.transform = transform

        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Carrega CSV
        self.annotations = pd.read_csv(annotation_file)

        # Lista dos arquivos .pt válidos
        self.file_list = sorted(glob(os.path.join(processed_dir, "*.pt")))
        if not self.file_list:
            raise ValueError(f"No .pt files found in {processed_dir}")

        # Filtra labels que têm .pt correspondente
        available_files = set(os.path.splitext(os.path.basename(f))[0] for f in self.file_list)

        self.file_to_label = {
            os.path.splitext(os.path.basename(row["audio_file"]))[0]: row["label"]
            for _, row in self.annotations.iterrows()
            if os.path.splitext(os.path.basename(row["audio_file"]))[0] in available_files
        }

        # Garante que só vamos usar arquivos com label
        self.file_list = [
            f for f in self.file_list
            if os.path.splitext(os.path.basename(f))[0] in self.file_to_label
        ]

        if not self.file_list:
            raise ValueError("No matching annotated .pt files found.")

        # Codifica os labels
        self.le = LabelEncoder()
        self.le.fit(list(self.file_to_label.values()))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pt_path = self.file_list[idx]
        base_name = os.path.splitext(os.path.basename(pt_path))[0]

        # Carrega espectrograma
        spectrogram = torch.load(pt_path)
        if not isinstance(spectrogram, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(spectrogram)}")

        # Codifica o label
        label_raw = self.file_to_label[base_name]
        label = self.le.transform([label_raw])[0]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return {
            "spectrogram": spectrogram,
            "label": label,
            "file_path": pt_path
        }
