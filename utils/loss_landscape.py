#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/09/13 08:55:08

import sys
import os
import time
import math
import copy
import importlib
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.status import progress_bar


def eval_loss(model, dataset, di, s, dir_num, steps):
    model.eval()
    losses = np.zeros(dataset.N_TASKS, dtype=np.float32)
    all_train_loaders = dataset.data_loaders['all_train']
    all_epoch = dir_num * len(steps)
    cur_epoch = di * len(steps) + s + 1
    with torch.no_grad():
        for i, loader in enumerate(all_train_loaders):
            for j, (bx, by, _) in enumerate(loader):

                bx = bx.to(model.device)
                by = by.to(model.device)

                outputs = model(bx)
                lss = F.mse_loss(outputs, by)
                losses[i] += lss.item()

                progress_bar(j, len(loader), 0,
                             f'{cur_epoch}/{all_epoch} - {i + 1}/{len(all_train_loaders)}',
                             lss.item())

            losses[i] /= len(loader)

    return losses

# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.
        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    else:
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())


# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
def create_random_direction(weights, ignore='biasbn', norm='filter', model=None):
    """
        Setup a random (normalized) direction with the same dimension as the weights.
        Args:
          weights: the given trained model
          ignore: 'biasbn', ignore biases and BN parameters.
        Returns:
          direction: a random direction with the same dimension as weights.
    """

    # random direction
    direction = []
    for w in weights:
        d = torch.randn(w.size())
        d = d.to(model.device)
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                # keep directions for weights/bias that are only 1 per node
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)

        direction.append(d)

    return direction


# https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py
def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        for (p, w, d) in zip(net.parameters(), weights, directions):
            p.data = w + d * step

    return net


def calculate_loss(model, dataset, steps, dir_num=10, output_dir=None):

    train_lss = np.zeros(
        (dataset.N_TASKS, dir_num, len(steps)), dtype=np.float32)

    with torch.no_grad():
        # calculate train_loss and train_acc
        trained_weights = copy.deepcopy(list(model.net.parameters()))

        for di in range(dir_num):
            torch.manual_seed(1024 + di)
            direction = create_random_direction(
                model.net.parameters(), model=model)
            for s, step in enumerate(steps):
                model.net = set_weights(
                    model.net, trained_weights, direction, step)

                train_lss[:, di, s] = eval_loss(
                    model, dataset, di, s, dir_num, steps)

        print()
        set_weights(model.net, trained_weights)

    # save the training loss
    npz_filename = '%s/loss_landscape/1d_loss_%s.npz' % (
        output_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(os.path.dirname(npz_filename), exist_ok=True)
    np.savez(npz_filename, train_loss=train_lss)

    print(f"Saved the training loss to {npz_filename}")

    return train_lss


def plot_1d_loss_all(loss, steps, output_dir, file_name=None, show=False):

    if file_name is None:
        file_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    loss = np.array(loss).unsqueeze(0)
    # loss = n_tasks * n_tasks * direction_num * steps

    print("train_loss:")
    print(loss)

    save_lss = np.ones((loss.reshape((-1, 1)).shape[0], 5))
    r = 0

    # loss map
    fig, axes = plt.subplots(nrows=loss.shape[0], ncols=loss.shape[1], sharex='all', sharey='all',
                             figsize=(loss.shape[0] * 2.5, loss.shape[1] * 2.5))

    for i in range(loss.shape[0]):
        axes[i, 0].set_ylim(0, 5)
        for j in range(loss.shape[1]):
            for k in range(loss.shape[2]):
                axes[i, j].plot(steps, loss[j, i, k], 'b-', linewidth=1)
                axes[i, j].set_title('Task %d' % j)
                axes[i, j].set_ylabel('loss of Task %d' % i)

                for m, s in enumerate(steps):
                    save_lss[r, 0] = j
                    save_lss[r, 1] = i
                    save_lss[r, 2] = k
                    save_lss[r, 3] = s
                    save_lss[r, 4] = loss[j, i, k, m]
                    r += 1

    for ax in axes.flat:
        ax.set(xlabel='Disturbance')
        ax.label_outer()

    plt.tight_layout()

    figname = '%s/loss_landscape/1d_loss_%s.pdf' % (output_dir, file_name)
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)

    csv_filename = '%s/loss_landscape/1d_loss_%s.csv' % (output_dir, file_name)
    csv_file = open(csv_filename, 'ab')
    np.savetxt(csv_filename, save_lss)
    csv_file.close()

    print(f"Saved the loss landscape to {figname} and {csv_filename}")

    if show:
        plt.show()
