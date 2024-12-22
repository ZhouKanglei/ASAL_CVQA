# -*- coding: utf-8 -*-
# @Time: 2023/6/20 22:43
import os

import numpy as np
import torch

from datasets import get_dataset
from models import get_model

from utils.misc import init_seed, count_param


class Processor(object):
    def __init__(self, args):
        self.args = args
        # init seed
        self.init_seed()
        # load dataset
        self.load_dataset()
        # load model
        self.load_model()

    def init_seed(self):
        if self.args.seed is not None:
            self.args.logging.info(f'Initialize seed with {self.args.seed}')
            init_seed(self.args.seed)

    def load_model(self):
        backbone = self.dataset.get_backbone()
        loss = self.dataset.get_loss(loss_type=self.args.loss_type,
                                     mse_loss_weight=self.args.mse_loss_weight,
                                     plcc_loss_weight=self.args.plcc_loss_weight)
        self.model = get_model(self.args, backbone, loss,
                               self.dataset.get_transform())
        self.args.logging.info(f'Load model: {self.args.model} ('
                               f'{count_param(self.model.net.feature_extractor):,d} + '
                               f'{count_param(self.model.net.regressor):,d} = '
                               f'{count_param(self.model.net):,d})')

        self.model.net = self.model.net.to(self.args.output_device)
        self.net = self.model.net
        self.model.module = self.model.net.module if hasattr(
            self.model.net, 'module') else self.model.net

        # load pretrain model
        if self.args.pretrain:
            if os.path.exists(self.args.weight_path):
                state = torch.load(self.args.weight_path)
                self.model.load_state_dict(state['model'])
                self.args.logging.info(
                    f'Load pretrained model state from {self.args.weight_path}.')
            else:
                self.args.logging.info(
                    f'No pretrained model found at {self.args.weight_path}.')

    def load_dataset(self):
        self.dataset = get_dataset(args=self.args)
        self.args.logging.info(
            f'Load dataset: {self.args.dataset} ({self.dataset.SETTING})')
