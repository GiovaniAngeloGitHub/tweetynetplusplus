import torch
import torch.nn.functional as F

from tweetynetplusplus.config import settings

class TemporalPadCrop:
    """
    A transform that pads or crops the temporal dimension of a spectrogram to a target width.
    
    Args:
        target_width (int): The target width (time steps) for the output spectrogram.
                          Defaults to settings.data.target_width.
    """
    def __init__(self, target_width: int = settings.data.target_width):
        self.target_width = target_width

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the temporal padding or cropping to the input tensor.
        
        Args:
            tensor (torch.Tensor): Input spectrogram tensor of shape (C, H, W)
            
        Returns:
            torch.Tensor: Padded or cropped tensor with shape (C, H, target_width)
        """
        try:
            _, h, w = tensor.shape
            
            if w < self.target_width:
                # Pad the right side of the spectrogram
                pad = self.target_width - w
                return F.pad(tensor, (0, pad))
            elif w > self.target_width:
                # Crop the right side of the spectrogram
                return tensor[:, :, :self.target_width]
                
            return tensor
            
        except Exception as e:
            raise RuntimeError(f"Error in TemporalPadCrop: {str(e)}")


class NormalizeTensor:
    """
    A transform that normalizes a tensor by subtracting the mean and dividing by the standard deviation.
    Adds a small epsilon to the standard deviation to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor of any shape
            
        Returns:
            torch.Tensor: Normalized tensor with zero mean and unit variance
        """
        try:
            # Calculate mean and std
            mean = tensor.mean()
            std = tensor.std() + self.eps  # Add epsilon to avoid division by zero
            
            if std < 1e-5:  # Check for very small standard deviation
                pass
                
            return (tensor - mean) / std
            
        except Exception as e:
            raise RuntimeError(f"Error in NormalizeTensor: {str(e)}")
