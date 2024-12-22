#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/08/20 13:38:05
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAE(nn.Module):

    def __init__(self, in_channels=512, out_channels=1):
        super(DAE, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fch = nn.Linear(256, 128)
        self.fc2_mean = nn.Linear(128, out_channels)
        self.fc2_logvar = nn.Linear(128, out_channels)

    def encode(self, x):
        h0 = F.relu(self.fc1(x))
        h1 = F.relu(self.fch(h0))
        mu = self.fc2_mean(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        esp = torch.randn_like(std).to(mu.device)
        z = mu + std * esp if self.training else mu
        return z

    def forward(self, x):
        x_size = x.size()
        h = x.view(x_size[0] * x_size[1], -1)

        mu, logvar = self.encode(h)
        z = self.reparametrize(mu, logvar)

        z = z.view(x_size[0], x_size[1]).mean(dim=1)

        return z


class MLP(nn.Module):

    def __init__(self, in_channels=512, out_channels=1):
        super(MLP, self).__init__()

        middle_channels = 128

        self.mlp1 = nn.Linear(in_channels, middle_channels)
        self.mlp2 = nn.Linear(middle_channels, out_channels)

    def encode(self, x):
        h0 = self.mlp1(x)
        h1 = self.mlp2(h0)

        return h1

    def forward(self, x):

        x_size = x.size()
        h = x.view(x_size[0] * x_size[1], -1)

        z = self.encode(h)

        z = z.view(x_size[0], x_size[1]).mean(dim=1)

        return z
