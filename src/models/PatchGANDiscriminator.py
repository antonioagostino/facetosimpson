import torch.nn as nn

class PatchGANDiscriminator(nn.Module):

    def __init__(self, bias: bool, normaliz_layer: nn.Module):
        """Constructs a PatchGAN discriminator (Pix2Pix Discriminator)
    
            Parameters:
                bias (bool): Use bias in the convolutional layers
                normaliz_layer (nn.Module): Normalization layer to use
        """
        super().__init__()
        model_blocks = [
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=bias),
            normaliz_layer(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=bias),
            normaliz_layer(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=bias),
            normaliz_layer(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1, bias=bias)
        ]

        self.model = nn.Sequential(*model_blocks)

    def forward(self, x):
        return self.model(x)