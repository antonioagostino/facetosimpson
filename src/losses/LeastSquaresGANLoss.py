import torch.nn as nn
import torch

class LeastSquaresGANLoss(nn.Module):
    """
    A class that implements the loss for the Least Squares GAN.
    """
    def __init__(self, device: torch.device):
        """
            Initializes the loss.
            :param device: the device on which the loss is computed
        """
        super().__init__()
        self.device = device
        self.loss = nn.MSELoss()

    def __call__(self, prediction: torch.Tensor, real_label: bool):
        if real_label is True:
            real_label_tensor = torch.tensor(1.0).to(self.device)
        else:
            real_label_tensor = torch.tensor(0.0).to(self.device)
        
        real_label_tensor = real_label_tensor.expand_as(prediction)

        return self.loss(prediction, real_label_tensor)
