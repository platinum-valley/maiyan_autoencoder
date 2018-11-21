import numpy as np
import torch
from torch import nn

class Classifier(nn.Module):

    def __init__(self, label_num):
        super(Classifier, self).__init__()

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

        self.comp = nn.Sequential(
                nn.Conv2d(256, 10, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        self.fc1 = nn.Sequential(
                nn.Linear(640, 160),
            )

        self.fc2 = nn.Linear(160, label_num)

        self.convolution = nn.Sequential(
                self.conv1,
                self.conv2,
                self.conv3,
                self.conv4,
                self.comp
            )

    def forward(self, x):
        x = self.convolution(x)
        x = x.view(x.size()[0], -1)
        pred = self.fc2(self.fc1(x))
        return pred

