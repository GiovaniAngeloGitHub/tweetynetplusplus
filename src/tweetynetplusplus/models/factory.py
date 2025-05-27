import importlib
import timm
import torch.nn as nn

def get_model_from_config(config_model: dict, num_classes: int):
    if config_model.get("custom", False):
        # Importação de modelo customizado
        module_path = config_model["module_path"]
        class_name = config_model["class_name"]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(num_classes=num_classes)

    # Fallback para modelos do TIMM
    model_name = config_model["name"]
    pretrained = config_model.get("pretrained", True)
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
