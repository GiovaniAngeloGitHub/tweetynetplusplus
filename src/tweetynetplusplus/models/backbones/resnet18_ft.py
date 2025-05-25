import timm
import torch.nn as nn

class ResNet18Finetune(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        ResNet18 model with a custom classifier for fine-tuning.
        
        Args:
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Load pre-trained ResNet18
        self.backbone = timm.create_model("resnet18", 
                                         pretrained=True, 
                                         num_classes=0,  # Remove the default classifier
                                         in_chans=3)      # 3 channels for stacked spectrograms
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, num_classes)
        )
        


    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        if self.training:
            pass
        
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Classification head
        logits = self.classifier(features)
        
        return logits
