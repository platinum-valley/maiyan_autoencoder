import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.batch_size = 16
        self.num_epoch = 10
        self.learning_rate = 0.001

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Conv2d(256, 10, kernel_size=3, stride=2, padding=1)


        self.encoder = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1
        )
        """
        self.fc22 = nn.Linear(
            nn.Linear(2, 512*8*8, bias=True),
            #nn.ReLU()
        )
        """
        self.fc2 = nn.Sequential(
            nn.ConvTranspose2d(10, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.conv4dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.conv3dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.conv2dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.conv1dec = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2)
        )

        self.decoder = nn.Sequential(
            self.fc2,
            self.conv4dec,
            self.conv3dec,
            self.conv2dec,
            self.conv1dec
        )

    def forward(self, x):
        #encoded = self.encoder(x)
        encoded = self.conv1(x)
        encoded = self.conv2(encoded)
        encoded = self.conv3(encoded)
        encoded = self.conv4(encoded)
        encoded = self.fc1(encoded)
        decoded = self.fc2(encoded)
        decoded = self.conv4dec(decoded)
        decoded = self.conv3dec(decoded)
        decoded = self.conv2dec(decoded)[:, :, 1:-2, 1:-2]
        decoded = self.conv1dec(decoded)[:, :, 1:-2, 1:-2]
        decoded = nn.Sigmoid()(decoded)

        return encoded, decoded

