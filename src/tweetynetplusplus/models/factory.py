import timm
import torch.nn as nn

def get_model_from_config(model_name: str, num_classes: int, pretrained: bool = True):
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
    except Exception as e:
        raise ValueError(f"Modelo '{model_name}' não encontrado no timm: {str(e)}")

    if hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise NotImplementedError(f"O modelo '{model_name}' não tem um atributo 'classifier' ou 'fc' reconhecível.")

    return model
