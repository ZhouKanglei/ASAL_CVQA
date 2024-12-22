# -*- coding: utf-8 -*-
# @Time: 2023/6/20 20:33
import argparse
import datetime
import logging
import os
import json
import pprint

import torch
import yaml
import ast

from utils.misc import cmdAction, str2bool, copy_dir
from utils.clean_empty_logs import clean_empty_logs


from datasets import NAMES as DATASET_NAMES
from models import get_all_models


class Parser(object):

    def __init__(self):
        # arguments parsing
        self.get_parser()
        # read configuration
        self.get_config()
        # merge configuration and parser parameters
        self.merge_args()
        # check arguments
        self.check_args()
        # get logging
        self.get_log()
        # save files and dirs
        if self.args.local_rank <= 0:
            self.back_proj()
        # print args
        self.args.logging.info(
            'Args: \n' + pprint.pformat(vars(self.args)))

    def get_parser(self):
        parser = argparse.ArgumentParser(description='CL for AQA')
        # config file
        parser.add_argument('--config', type=str, required=True,
                            help='Config name.', action=cmdAction)
        parser.add_argument(
            '--exp_name', type=str, default='default', help='Work dir.', action=cmdAction)
        parser.add_argument('--phase', type=str, default='train',
                            help='train or test', action=cmdAction)
        parser.add_argument('--pretrain', type=str2bool,
                            default=False, help='pretrain', action=cmdAction)
        parser.add_argument('--weight_path', type=str, default=None,
                            help='pretrain weight', action=cmdAction)
        parser.add_argument('--base_pretrain', type=str2bool, default=False,
                            help='base session pretrain', action=cmdAction)
        parser.add_argument('--base_pretrain_model_path', type=str,
                            default=None, help='dist ddp', action=cmdAction)
        parser.add_argument('--local_rank', type=int,
                            default=-1, help='dist ddp', action=cmdAction)
        parser.add_argument(
            '--user', type=str, default=os.environ['USER'], help='user', action=cmdAction)
        # experimental arguments
        parser.add_argument('--dataset', type=str, choices=DATASET_NAMES, required=False,
                            help='Which dataset to perform experiments on.', action=cmdAction)
        parser.add_argument('--model', type=str, choices=get_all_models() + ['base', 'joint_wo_base'],
                            help='Model name.', action=cmdAction)
        parser.add_argument('--lr', type=float, default=0.1,
                            help='Learning rate.', action=cmdAction)
        parser.add_argument('--loss_type', type=str, default='mse', choices=[
                            'mse', 'plcc', 'mix'], help='Loss type.', action=cmdAction)
        parser.add_argument('--mse_loss_weight', type=float, default=1,
                            help='MSE loss weight.', action=cmdAction)
        parser.add_argument('--plcc_loss_weight', type=float, default=1,
                            help='PLCC loss weight.', action=cmdAction)

        parser.add_argument('--optim_wd', type=float, default=0.,
                            help='optimizer weight decay.', action=cmdAction)
        parser.add_argument('--optim_mom', type=float, default=0.,
                            help='optimizer momentum.', action=cmdAction)
        parser.add_argument('--optim_nesterov', type=int, default=0,
                            help='optimizer nesterov momentum.', action=cmdAction)
        # elarly stop
        parser.add_argument('--early_stop', type=str2bool, default=False,
                            help='Enable early stop.', action=cmdAction)
        parser.add_argument('--patience', type=int, default=10,
                            help='Early stop patience.', action=cmdAction)

        parser.add_argument('--n_epochs', type=int, default=20,
                            help='Training epoch.', action=cmdAction)
        parser.add_argument('--n_tasks', type=int, default=5,
                            help='Task number.', action=cmdAction)
        parser.add_argument('--batch_size', type=int,
                            default=128, help='Batch size.', action=cmdAction)
        parser.add_argument('--num_workers', type=int, default=16,
                            help='Dataset workers.', action=cmdAction)

        parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                            help='GPU id list.', action=cmdAction)

        parser.add_argument('--seed', type=int, default=1024,
                            help='The random seed.', action=cmdAction)
        parser.add_argument('--train_ratio', type=float,
                            default=0.7, help='Train ratio', action=cmdAction)
        # number of key frames
        parser.add_argument('--num_key_frames', type=int, default=2,
                            help='Number of key frames.', action=cmdAction)

        # model arguments
        parser.add_argument('--regressor', type=str,
                            help='Regressor model', action=cmdAction)
        parser.add_argument(
            '--regressor_args', type=ast.literal_eval, default=None, action=cmdAction)

        parser.add_argument('--projector', type=str,
                            help='Projector model', action=cmdAction)
        parser.add_argument(
            '--projector_args', type=ast.literal_eval, default=None, action=cmdAction)

        parser.add_argument('--adapter', type=str,
                            help='Adapter model for ours', action=cmdAction)
        parser.add_argument(
            '--adapter_args', type=ast.literal_eval, default=None, action=cmdAction)

        parser.add_argument('--feature_extractor', type=str,
                            help='Feature extractor model', action=cmdAction)
        parser.add_argument(
            '--feature_extractor_args', type=ast.literal_eval, default=None, action=cmdAction)

        # Management args
        parser.add_argument('--non_verbose', default=False, type=str2bool,
                            help='Make progress bars non verbose.', action=cmdAction)
        parser.add_argument('--disable_log', default=False, type=str2bool,
                            help='Enable csv logging.', action=cmdAction)

        parser.add_argument('--validation', default=False, type=str2bool,
                            help='Test on the validation set.', action=cmdAction)
        parser.add_argument('--ignore_other_metrics', default=False, type=str2bool,
                            help='disable additional metrics.', action=cmdAction)
        parser.add_argument('--debug_mode', default=False, type=str2bool,
                            help='Run only a few forward steps per epoch', action=cmdAction)
        parser.add_argument('--fewshot', default=False, type=str2bool,
                            help='Few-shot mode', action=cmdAction)
        parser.add_argument('--fewshot_num', type=int, default=5 * 4,
                            help='Few-shot number', action=cmdAction)

        # Rehearsal args
        parser.add_argument('--buffer_size', type=int, default=10,
                            help='The size of the memory buffer.', action=cmdAction)
        parser.add_argument('--minibatch_size', type=int, default=10,
                            help='The batch size of the memory buffer.', action=cmdAction)
        parser.add_argument('--alpha', type=float,
                            default=1, help='ER penalty weight.', action=cmdAction)
        parser.add_argument('--beta', type=float, default=1,
                            help='ER penalty weight.', action=cmdAction)
        # EWC parameters
        parser.add_argument('--e_lambda', type=float, default=1,
                            help='lambda weight for EWC', action=cmdAction)
        parser.add_argument('--gamma', type=float, default=1,
                            help='gamma parameter for EWC online', action=cmdAction)
        parser.add_argument('--batch_num', type=int, default=1,
                            help='Number of batches extracted from the buffer.', action=cmdAction)
        parser.add_argument('--num_samples_per_task', type=int, default=20,
                            help='Number of samples per task.', action=cmdAction)
        parser.add_argument('--noise_level', type=int, default=10,
                            help='Noise level.', action=cmdAction)

        # visualization of loss landscape
        parser.add_argument('--visual_landscape', default=True,
                            action='store_true', help='')
        parser.add_argument('--step_min', type=float, default=-1.0)
        parser.add_argument('--step_max', type=float, default=1.0)
        parser.add_argument('--step_size', type=float, default=0.04)
        parser.add_argument('--dir_num', type=int, default=1)

        self.args = parser.parse_args()

    def get_config(self):
        with open(self.args.config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def get_log(self):
        if self.args.local_rank > 0:
            logger = logging.getLogger(name=__name__)
            logger.propagate = False
            self.args.logging = logger
            return

        # logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger(self.args.proj_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s', "%y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_fh = logging.FileHandler(self.args.log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s', "%y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        self.args.logging = logger

        # logging info
        self.args.logging.info(f'New {self.args.log_file}.')

    def merge_args(self):
        for k, v in self.config.items():
            if k not in vars(self.args).keys():
                setattr(self.args, k, v)
            elif not hasattr(self.args, f'{k}_non_default'):
                setattr(self.args, k, v)

    def check_args(self):
        self.args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # default exp name
        self.args.proj_name = os.path.abspath(__file__).split('/')[-3]

        if not hasattr(self.args, f'user_non_default'):
            self.args.user = os.environ['USER']

        if self.args.exp_name == 'default':
            self.args.exp_name = self.args.model + f'{self.args.n_tasks:02d}'

            if self.args.weight_path is not None and 'joint' in self.args.model:
                self.args.exp_name = self.args.model + \
                    f'{self.args.n_tasks:02d}-pretrain_backbone'

            param_groups = {'regressor': f'-{self.args.regressor.split(".")[-1]}',
                            'alpha': f'-alpha{self.args.alpha}',
                            'beta': f'-beta{self.args.beta}',
                            'gamma': f'-gamma{self.args.gamma}',
                            'e_lambda':  f'-lambda{self.args.e_lambda}',
                            'buffer_size': f'-buffer{self.args.buffer_size}',
                            'batch_num': f'-bn{self.args.buffer_size}',
                            'num_samples_per_task': f'-{self.args.num_samples_per_task}_samples_per_task',
                            'noise_level': f'-noise_{self.args.noise_level}',
                            'base_pretrain': '-base' if self.args.base_pretrain else '',
                            'loss_type': f'-{self.args.loss_type}',
                            'mse_loss_weight': f'{self.args.mse_loss_weight}',
                            'num_key_frames': f'-kf{self.args.num_key_frames}',
                            }

            for k, v in param_groups.items():
                if hasattr(self.args, f'{k}_non_default'):
                    self.args.exp_name += v

        # add user and version to the exp name
        exp_root_name = self.args.user + f'-seed{self.args.seed}'
        self.args.exp_dir = os.path.join(
            './outputs', exp_root_name +
            f'-fscl{self.args.fewshot_num}' if self.args.fewshot else exp_root_name,
            self.args.dataset if 'joint' not in self.args.model else '',
            self.args.exp_name
        )

        os.makedirs(self.args.exp_dir, exist_ok=True)
        # key frames
        if self.args.adapter_args is not None:
            if 'in_frames' in self.args.adapter_args.keys():
                self.args.num_key_frames = self.args.adapter_args['in_frames']

        # logger file and dir
        self.args.log_dir = os.path.join(self.args.exp_dir, 'logs')
        os.makedirs(self.args.log_dir, exist_ok=True)
        self.args.recyle_bin = os.path.join(self.args.log_dir, 'recycle_bin')
        # clean_empty_logs(self.args.log_dir, self.args.recyle_bin)

        self.args.log_file = os.path.join(self.args.log_dir,
                                          f'{self.args.phase}-{self.args.timestamp}.log')

        # gpus
        self.args.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.args.output_device = torch.device(f"cuda:{self.args.gpus[0]}")

        # train ratio
        if hasattr(self.args, 'train_ratio_non_default'):
            self.args.dataset_args['train_ratio'] = self.args.train_ratio

        # model
        if self.args.pretrain and self.args.weight_path == None:
            self.args.weight_path = os.path.join(
                self.args.exp_dir, f'weights/best.pth')

        if self.args.base_pretrain_model_path == None:
            self.args.base_pretrain_model_path = os.path.join(
                'weights', self.args.dataset + f'{self.args.n_tasks:02d}.pth')

        # output save dir for an experiment
        self.args.output_dir = os.path.join(self.args.exp_dir, f'outputs')
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.args.output_score_dir = os.path.join(
            self.args.output_dir, 'score')
        os.makedirs(self.args.output_score_dir, exist_ok=True)
        self.args.output_model_weight_dir = os.path.join(
            self.args.output_dir, 'weights')
        os.makedirs(self.args.output_model_weight_dir, exist_ok=True)
        self.args.output_result_dir = os.path.join(
            self.args.output_dir, 'results')
        os.makedirs(self.args.output_result_dir, exist_ok=True)

        # batch size
        if not hasattr(self.args, 'minibatch_size_non_default'):
            self.args.minibatch_size = self.args.batch_size

        # projector
        if 'gr' not in self.args.model:
            self.args.projector = None

    def reuse_config(self):
        config = os.path.join(
            self.args.exp_dir, f'configs/{self.args.phase}.yaml')
        os.makedirs(os.path.dirname(config), exist_ok=True)
        # load pre-trained configuration
        if self.args.phase == 'train' and self.args.pretrain:
            if not os.path.exists(config):
                return
            self.args.logging.info(f'Reuse existing {config}.')
            with open(config) as f:
                old_args = yaml.load(f, Loader=yaml.Loader)

            for k, v in old_args.items():
                if k in vars(self.args).keys() and not hasattr(self.args, f'{k}_non_default'):
                    if k not in ['local_rank', 'weight_path']:
                        if type(v) is str:
                            if 'test' not in v and 'train' not in v:
                                setattr(self.args, k, v)
                        else:
                            setattr(self.args, k, v)

        if self.args.phase == 'test':
            train_config = os.path.join(
                self.args.exp_dir, f'configs/train.yaml')
            self.args.logging.info(f'Reuse training {train_config}.')
            if not os.path.exists(train_config):
                return
            with open(train_config) as f:
                train_args = yaml.load(f, Loader=yaml.Loader)

            for k, v in train_args.items():
                if k in vars(self.args).keys() and not hasattr(self.args, f'{k}_non_default'):
                    if type(v) is str:
                        if 'test' not in v and 'train' not in v:
                            setattr(self.args, k, v)
                    else:
                        setattr(self.args, k, v)

    def back_proj(self):
        # save the configuration
        config = os.path.join(
            self.args.exp_dir, f'configs/{self.args.phase}.yaml')
        os.makedirs(os.path.dirname(config), exist_ok=True)

        with open(config, 'w') as f:
            yaml.dump(vars(self.args), f)
            self.args.logging.info(f'Save {config}.')

        # save main file and dirs
        for dir in ['main.py', 'utils', 'datasets', 'models', 'backbone']:
            new_dir = os.path.join(self.args.exp_dir, dir)
            if os.path.exists(dir):
                copy_dir(dir, new_dir)


if __name__ == '__main__':
    parser = Parser()
