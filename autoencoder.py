import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


class Autoencoder(nn.Module):

    def __init__(self, label_num):
        super(Autoencoder, self).__init__()
        self.label_num = label_num

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Conv2d(256, 10, kernel_size=3, stride=2, padding=1)

        self.fc2 = nn.Linear(640, 2, bias=True)

        self.fc_mu = nn.Sequential(
            nn.Linear(640, 200, bias=True),
            nn.LeakyReLU()
        )

        self.fc_var = nn.Sequential(
            nn.Linear(640, 200, bias=True),
            nn.ReLU()
        )

        self.emb_label = nn.Sequential(
            nn.Linear(self.label_num, 200, bias=True),
            nn.LeakyReLU()
        )

        self.fc2dec = nn.Linear(400, 640, bias=True)

        self.fc1dec = nn.Sequential(
            nn.ConvTranspose2d(10, 256, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.conv4dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU()

        )
        self.conv3dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.conv2dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.conv1dec = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )



    def reparameterize(self, mu, var):
        std = var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x, label):
        encoded = self.conv1(x)
        encoded = self.conv2(encoded)
        encoded = self.conv3(encoded)
        encoded = self.conv4(encoded)
        encoded = self.fc1(encoded).view(encoded.size()[0], -1)
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        emb_label = self.emb_label(label)
        z = self.reparameterize(mu, var)
        z = torch.cat((z, emb_label), 1)
        decoded = self.fc2dec(z)
        decoded = self.fc1dec(decoded.view(decoded.size()[0], 10, 8, 8))
        decoded = self.conv4dec(decoded)
        decoded = self.conv3dec(decoded)
        decoded = self.conv2dec(decoded)
        decoded = self.conv1dec(decoded)
        decoded = torch.clamp(decoded, min=0.0, max=1.0)
        return mu, var, decoded

    def encode(self, x):
        x = x.view(1, x.size()[0], x.size()[1], x.size()[2])
        encoded = self.conv1(x)
        encoded = self.conv2(encoded)
        encoded = self.conv3(encoded)
        encoded = self.conv4(encoded)
        encoded = self.fc1(encoded).view(encoded.size()[0], -1)
        mu = self.fc_mu(encoded)
        var = self.fc_var(encoded)
        return mu, var

    def generate(self, mu, var, label):
        z = self.reparameterize(mu, var).view(1, -1)
        emb_label = self.emb_label(label)
        z = torch.cat((z, emb_label), 1)
        decoded = self.fc2dec(z)
        decoded = self.fc1dec(decoded.view(decoded.size()[0], 10, 8, 8))
        decoded = self.conv4dec(decoded)
        decoded = self.conv3dec(decoded)
        decoded = self.conv2dec(decoded)
        decoded = self.conv1dec(decoded)
        decoded = torch.clamp(decoded, min=0.0, max=1.0)
        return decoded

