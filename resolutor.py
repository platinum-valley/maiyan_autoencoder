import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


class Resolutor(nn.Module):

    def __init__(self):
        super(Resolutor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Conv2d(256, 10, kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(640, 400, bias=True),
            nn.ReLU()
        )


        self.fc2dec = nn.Linear(400, 640, bias=True)

        self.fc1dec = nn.Sequential(
            nn.ConvTranspose2d(10, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.conv4dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.conv3dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.conv2dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.conv1dec = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2),
            nn.Sigmoid()
        )



    def forward(self, x):
        #encoded = self.encoder(x)
        encoded = self.conv1(x)
        encoded = self.conv2(encoded)
        encoded = self.conv3(encoded)
        encoded = self.conv4(encoded)
        encoded = self.fc1(encoded).view(encoded.size()[0], -1)
        hidden = self.fc(encoded)
        decoded = self.fc2dec(hidden)
        decoded = self.fc1dec(decoded.view(decoded.size()[0], 10, 8, 8))
        decoded = self.conv4dec(decoded)
        decoded = self.conv3dec(decoded)
        decoded = self.conv2dec(decoded)[:, :, 1:-2, 1:-2]
        decoded = self.conv1dec(decoded)[:, :, 1:-2, 1:-2]
        return decoded
