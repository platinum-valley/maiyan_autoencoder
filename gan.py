import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

class Discriminator(nn.Module):


    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(nn.Linear(65536, 1, bias=True),
            nn.Sigmoid()
        )

        self.discriminate = nn.Sequential(self.conv1,
            self.conv2,
            self.conv3,
            self.conv4
        )

    def forward(self, x):
        x = self.discriminate(x).view(x.size()[0], -1)
        x = self.fc1(x)
        return x
