#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/8 上午9:37

import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):

    def __init__(self, in_channels):
        super(Projector, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, in_channels)

    def forward(self, x):
        x_size = x.size()
        h = x.view(x_size[0] * x_size[1], -1)

        h1 = F.relu(self.fc1(h))
        y = F.relu(self.fc2(h1)) + h

        y = y.view(x_size)

        return y
