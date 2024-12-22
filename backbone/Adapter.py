#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/8 上午9:37
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_channels, 256)
        self.fc2 = nn.Linear(256, in_channels)

    def forward(self, x):
        x_size = x.size()
        h = x.view(x_size[0] * x_size[1], -1)

        h1 = F.relu(self.fc1(h))
        y = F.relu(self.fc2(h1)) + h

        y = y.view(x_size)

        return y


class AdapterT(nn.Module):

    def __init__(self, in_channels, out_channels, in_frames, out_frames):
        super(AdapterT, self).__init__()

        self.T = nn.Parameter(
            torch.ones(in_frames, out_frames)
        )

    def forward(self, x):
        T = torch.softmax(self.T, dim=0)
        h1 = torch.einsum('bcd, ce->bed', x, T)
        y = h1.contiguous()
        return y
