from torch.utils.data import WeightedRandomSampler
import os
from collections import Counter

def create_weighted_sampler(dataset, subset_indices):
    """
    Cria um WeightedRandomSampler baseado nas frequências das classes em um subconjunto do dataset.
    
    Args:
        dataset: instância de BirdsongSpectrogramDataset
        subset_indices: índices dos exemplos usados no conjunto de treino
    
    Returns:
        sampler: instância de WeightedRandomSampler
    """
    # Obtem os rótulos do subconjunto de treino
    subset_labels = []
    for idx in subset_indices:
        file_path = dataset.file_list[idx]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        label = dataset.file_to_label[base_name]
        subset_labels.append(label)

    label_counts = Counter(subset_labels)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [class_weights[label] for label in subset_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

