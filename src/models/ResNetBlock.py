import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, conv_channels: int, bias: bool, normaliz_layer: nn.Module):
        """ResNet Block
        A ResNet block is a is a convolutional block with a skip connection.
        Original ResNet paper: https://arxiv.org/pdf/1512.03385.pdf

        Parameters:
            conv_channels (int)         -- number of input and output channels of the Convolutional Layer.
            bias (bool)                 -- if the Convolutional Layer uses bias or not
            normaliz_layer (nn.Module)  -- Layer used to Normalize feature maps.
        """
        super().__init__()
        self.convolutional_blocks = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=0, bias=bias),
            normaliz_layer(conv_channels), 
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=0, bias=bias),
            normaliz_layer(conv_channels)
        )

    def forward(self, x):
        """Skip connection"""
        return x + self.convolutional_blocks(x)