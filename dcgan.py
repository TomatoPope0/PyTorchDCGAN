import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_noises=100, num_colors=3, num_depth=128, image_size=64):
        super(Generator, self).__init__()

        if image_size % 16 != 0:
            raise Exception("Size of the image must be divisible by 16")
        self.image_size = image_size

        # TODO: Maybe we can use ConvTranspoed2d instead
        self.lin = nn.Linear(num_noises, num_depth * 8 * image_size * image_size)
        # Some calculations behind deciding kernel size, stride, padding:
        # (x+2p-k)/s+1=x/2 <=> s != 0 or 2, k = ... or p = k/2 - 1, s = 2
        # => Choose k = 4, s = 2, p = 1
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(num_depth * 8, num_depth * 4, 4, 2, 1),
            nn.BatchNorm2d(num_depth * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_depth * 4, num_depth * 2, 4, 2, 1),
            nn.BatchNorm2d(num_depth * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_depth * 2, num_depth, 4, 2, 1),
            nn.BatchNorm2d(num_depth),
            nn.ReLU(),
            nn.ConvTranspose2d(num_depth, num_colors, 4, 2, 1),
            # Paper: Quick saturation -> coverage of the color space
            nn.Tanh()
        )
    
    def forward(self, x):
        lin_out = self.lin(x)
        conv_out = self.conv(lin_out.view(-1, self.image_size, self.image_size))
        return conv_out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x
