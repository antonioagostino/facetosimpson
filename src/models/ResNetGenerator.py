from .ResNetBlock import ResNetBlock
import torch.nn as nn

class ResNetGenerator(nn.Module):
    """ A ResNet Generator generates an image from another image.
        Between some Downsampling and Upsampling layers, there are
        N ResNetBlocks.
    """
    def __init__(self, num_resnet_blocks: int, bias: bool, normaliz_layer: nn.Module):
        super().__init__()
        model_blocks = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=bias),
            normaliz_layer(64),
            nn.ReLU(True)
        ]

        # Downsampling
        model_blocks.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias))
        model_blocks.append(normaliz_layer(128))
        model_blocks.append(nn.ReLU(True))
        model_blocks.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias))
        model_blocks.append(normaliz_layer(256))
        model_blocks.append(nn.ReLU(True))

        # ResNet Blocks
        for _ in range(num_resnet_blocks):
            model_blocks.append(ResNetBlock(256, bias, normaliz_layer))

        # Upsampling
        model_blocks.append(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, 
                            output_padding=1, bias=bias))
        model_blocks.append(normaliz_layer(128))
        model_blocks.append(nn.ReLU(True))
        model_blocks.append(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, 
                            output_padding=1, bias=bias))
        model_blocks.append(normaliz_layer(64))
        model_blocks.append(nn.ReLU(True))

        model_blocks.append(nn.ReflectionPad2d(3))
        model_blocks.append(nn.Conv2d(64, 3, kernel_size=7, padding=0))
        model_blocks.append(nn.Tanh())

        self.model = nn.Sequential(*model_blocks)

    def forward(self, x):
        return self.model(x)