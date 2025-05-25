class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    
    Args:
        patience (int): Number of epochs to wait before stopping when the loss
                      doesn't improve. Default: 5.
        min_delta (float): Minimum change in the monitored quantity to qualify as
                         an improvement. Default: 0.
        verbose (bool): If True, prints a message for each validation loss improvement.
                      Default: True.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        

    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped based on the validation loss.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        try:
            # Calculate improvement
            improvement = (self.best_loss - val_loss) / self.best_loss if self.best_loss != 0 else float('inf')
            
            if val_loss < self.best_loss - self.min_delta:
                # Improvement found
                self.best_loss = val_loss
                self.counter = 0
            else:
                # No improvement
                self.counter += 1
                
                if self.counter >= self.patience:
                    self.early_stop = True
            
            return self.early_stop
            
        except Exception as e:
            return False
