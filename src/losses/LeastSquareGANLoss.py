import torch.nn as nn
import torch

class LeastSquareGANLoss(nn.Module):
    def __init__(self, device: torch.device):
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
