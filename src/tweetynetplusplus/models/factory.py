
import importlib
import timm
import torch.nn as nn

def get_model_from_config(model_name: str, num_classes: int, pretrained: bool = True, import_path: str = None):
    if import_path:
        # Importação dinâmica do módulo e da classe
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(num_classes=num_classes)

    # Fallback para modelos do timm
    model = timm.create_model(model_name, pretrained=pretrained)

    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"O modelo '{model_name}' não possui atributo fc/classifier reconhecível.")

    return model
