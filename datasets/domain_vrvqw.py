#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/08/14 11:33:32

import glob
import os
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from backbone.VRQAMLP import AQAMLP
from datasets.utils.continual_dataset import ContinualDataset
from utils.misc import logging_info
from utils.misc import init_seed


class VRVQW_Dataset(torch.utils.data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, dataset_args, samples, transform, subset='all', aug=False):
        super(VRVQW_Dataset, self).__init__()
        # Load the video information
        video_info_file_path = dataset_args['video_info_file_path']
        video_info = pd.read_csv(video_info_file_path)
        # Select the video names and MOS scores based on the samples
        index = video_info['VideoName'].isin(samples)
        self.video_names = video_info.loc[index, 'VideoName'].tolist()
        self.mos_scores = video_info.loc[index, 'MOS'].tolist()

        # Variables
        self.video_dir = dataset_args['video_dir']
        self.length = len(self.video_names)
        self.transform = transform
        self.aug = aug

    def __len__(self):
        return self.length

    def load_video(self, video_name):
        # image selection step
        if video_name[:2] == '7_':
            step = 1
            video_name_str = video_name[2:]
        elif video_name[:3] == '15_':
            step = 2
            video_name_str = video_name[3:]
        else:
            raise ValueError('video name error')

        # image list
        image_list = sorted(
            (glob.glob(os.path.join(self.video_dir, video_name_str, '*.png')))
        )
        selected_image_list = []
        for _ in range(7):
            idx = _ * step + random.randint(0, 1) if self.aug else _ * step
            selected_image_list.append(image_list[idx])

        video = []
        for _ in selected_image_list:
            img = Image.open(_)
            img = self.transform(img)
            video.append(img)
        # cat the images
        try:
            video = torch.stack(video, dim=0)
        except Exception as e:
            print(e)
            print(video_name)

        return video

    def __getitem__(self, idx):

        video_name = self.video_names[idx]
        video_score = torch.FloatTensor(np.array(float(self.mos_scores[idx])))
        transformed_video = self.load_video(video_name)

        return transformed_video, video_score, video_name


def select_sample(samples_, num=20, seed=1024):
    random.seed(seed)
    samples = samples_.copy()
    selected_samples = []
    categories = ['7_A', '7_B', '15_A', '15_B']
    num_per_category = num // len(categories)
    for category in categories:
        category_samples = [
            sample for sample in samples if sample.startswith(category)]
        random.shuffle(category_samples)
        if num_per_category > len(category_samples):
            selected_samples += category_samples
            samples = [
                sample for sample in samples if sample not in selected_samples]
            if len(samples) < num - len(category_samples):
                continue
            random.shuffle(samples)
            selected_samples += samples[:num - len(category_samples)]

        else:
            selected_samples += category_samples[:num_per_category]
        samples = [sample for sample in samples if sample not in selected_samples]

    return selected_samples


def dataset_split(file_path, video_info_file_path,
                  train_ratio=0.8, seed=1024, logger=None,
                  fewshot=False, fewshot_num=20):
    datainfo = pd.read_csv(file_path)
    video_info = pd.read_csv(video_info_file_path)

    # Get unique video names from the dataset
    video_names = datainfo['video_names']

    # Get unique categories from the dataset
    categories = set(datainfo['category'].tolist())
    categories = list(categories)
    categories = sorted(categories)

    # Get out of distribution samples
    categories.pop(categories.index('Others'))
    ood_samples = video_names[datainfo['category'] == 'Others'].to_list()
    selected_ood_videonames = []
    all_videonames = video_info['VideoName'].tolist()

    for videoname in all_videonames:
        if videoname[4:] in ood_samples:
            selected_ood_videonames.append(videoname)
        elif videoname[5:] in ood_samples:
            selected_ood_videonames.append(videoname)

    # Number of tasks
    n_tasks = len(categories)
    logging_info(f'Number of tasks: {n_tasks}', logger)

    # Set the seed for reproducibility
    random.seed(seed)

    # Shuffle the categories
    logging_info(f'Categories: {categories}', logger)
    random.shuffle(categories)
    categories = ['CG', 'Landscape', 'Sports', 'Shows', 'Cityscape']
    logging_info(f'Categories: {categories}', logger)

    # Split the categories into training and testing sets
    train_sample_splits = []
    test_sample_splits = []
    stat_info = {
        'category': categories,
        'category_samples': [],
        'number_of_train_samples': [],
        'number_of_test_samples': [],
        'number_of_other_samples': 0
    }

    for _, category in enumerate(categories):
        # Get the videos belonging to the current session
        session_videos = video_names[datainfo['category'] == category].tolist()
        stat_info['category_samples'].append(len(session_videos))

        # Get all the videos from the video info file by the session videos
        selected_videonames = []
        all_videonames = video_info['VideoName'].tolist()

        for videoname in all_videonames:
            if videoname[4:] in session_videos:
                selected_videonames.append(videoname)
            elif videoname[5:] in session_videos:
                selected_videonames.append(videoname)

        # Split the videos into training and testing sets
        selected_videonames = pd.Series(selected_videonames)
        train_videos = selected_videonames.sample(
            frac=train_ratio, random_state=seed)
        test_videos = selected_videonames.drop(train_videos.index)

        # Add the training videos to the train_sample_splits list
        if fewshot:
            samples = train_videos.to_list()
            train_videos_ = select_sample(samples, num=fewshot_num, seed=seed)
            train_sample_splits.append(train_videos_)
        else:
            train_videos_ = train_videos.to_list()
            train_sample_splits.append(train_videos_)

        stat_info['number_of_train_samples'].append(len(train_videos_))

        # Add the testing videos to the test_samples list
        test_sample_splits.append(test_videos.to_list())
        stat_info['number_of_test_samples'].append(len(test_videos))

    stat_info['number_of_other_samples'] = len(ood_samples)

    # Print the statistics
    logging_info("CL split statistics:".center(
        36, ' ').center(80, '-'), logger)
    logging_info("\n" + json.dumps(stat_info, indent=4), logger)
    logging_info("".center(80, '-'), logger)

    return train_sample_splits, test_sample_splits, selected_ood_videonames, categories


def joint_dataset_split(video_info_file_path, train_ratio=0.7, seed=1024, logger=None):
    video_info = pd.read_csv(video_info_file_path)

    # Get all video names from the dataset
    all_videonames = video_info['VideoName']

    train_videonames = all_videonames.sample(
        frac=train_ratio, random_state=seed)
    test_videonames = all_videonames.drop(train_videonames.index)

    stat_info = {
        'number_of_train_samples': len(train_videonames),
        'number_of_test_samples': len(test_videonames),
        'train_ratio': train_ratio
    }

    # Print the statistics
    logging_info("Joint split statistics:".center(
        36, ' ').center(80, '-'), logger)
    logging_info("\n" + json.dumps(stat_info, indent=4), logger)
    logging_info("".center(80, '-'), logger)

    return train_videonames, test_videonames


class DomainVRVQW(ContinualDataset):
    NAME = 'domain-vrvqw'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 5

    def __init__(self, args):
        super(DomainVRVQW, self).__init__(args)
        # data splits
        self.train_sample_splits, self.test_sample_splits, \
            self.base_session_split, categories = dataset_split(
                file_path=self.args.dataset_args['category_file_path'],
                video_info_file_path=self.args.dataset_args['video_info_file_path'],
                train_ratio=self.args.train_ratio,
                logger=self.args.logging,
                fewshot=self.args.fewshot,
                fewshot_num=self.args.fewshot_num)

        self.joint_train_sample_split, self.joint_test_sample_split = joint_dataset_split(
            video_info_file_path=self.args.dataset_args['video_info_file_path'],
            train_ratio=self.args.train_ratio,
            seed=self.args.seed, logger=self.args.logging)

        self.N_TASKS = len(categories)
        # train and test transformation
        self.train_trans = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.test_trans = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # initialize data loaders
        self.data_loaders = {
            'base_train': self.get_base_data_loader(phase='train'),
            'base_test': self.get_base_data_loader(phase='test'),
            'all_train': self.get_all_data_loaders(phase='train'),
            'all_test': self.get_all_data_loaders(phase='test'),
            'joint_train_full': self.get_joint_data_loader_old(phase='train'),
            'joint_test_full': self.get_joint_data_loader_old(phase='test'),
            'joint_train': self.get_joint_data_loader(phase='train'),
            'joint_test': self.get_joint_data_loader(phase='test'),
            'task_idx': self.i
        }

        self.i = -1  # Task index

    def worker_init_fn(self, worker_id):
        init_seed(self.args.seed + worker_id)

    def get_data_loader(self, phase='train', subset='all', i=None):
        if i is None:
            i = self.i

        if phase == 'train':
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.train_trans,
                                    subset=subset, samples=self.train_sample_splits[i],
                                    aug=True)
        else:
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.test_trans,
                                    subset=subset, samples=self.test_sample_splits[i])

        loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True if phase == 'train' else False,
            worker_init_fn=self.worker_init_fn if phase == 'train' else None
        )

        return loader

    def get_base_data_loader(self, phase='train', subset='all'):

        dataset = VRVQW_Dataset(self.args.dataset_args,
                                transform=self.train_trans if phase == 'train' else self.test_trans,
                                subset=subset, samples=self.base_session_split,
                                aug=True if phase == 'train' else False)

        loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True if phase == 'train' else False,
            worker_init_fn=self.worker_init_fn if phase == 'train' else None
        )

        return loader

    def get_joint_data_loader(self, phase='train', subset='all'):
        if phase == 'train':
            samples = []
            for i in range(self.N_TASKS):
                samples += self.train_sample_splits[i]
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.train_trans,
                                    subset=subset, samples=samples, aug=True)
        else:
            samples = []
            for i in range(self.N_TASKS):
                samples += self.test_sample_splits[i]
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.test_trans,
                                    subset=subset, samples=samples)

        loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True if phase == 'train' else False,
            worker_init_fn=self.worker_init_fn if phase == 'train' else None
        )

        return loader

    def get_joint_data_loader_old(self, phase='train', subset='all', i=None):

        if phase == 'train':
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.train_trans,
                                    subset=subset, samples=self.joint_train_sample_split,
                                    aug=True)
        else:
            dataset = VRVQW_Dataset(self.args.dataset_args, transform=self.test_trans,
                                    subset=subset, samples=self.joint_test_sample_split)

        loader = DataLoader(
            dataset, batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True if phase == 'train' else False,
            worker_init_fn=self.worker_init_fn if phase == 'train' else None
        )

        return loader

    def get_all_data_loaders(self, phase='train', subset='all'):
        loaders = []
        for i in range(self.N_TASKS):
            loaders.append(self.get_data_loader(
                phase=phase, subset=subset, i=i))

        return loaders

    def get_observed_data_loaders(self, phase='train', subset='all'):
        loaders = []
        for i in range(self.i + 1):
            loaders.append(self.get_data_loader(
                phase=phase, subset=subset, i=i))

        return loaders

    def get_data_loaders(self):
        self.i += 1

        new_data_loaders = {
            'train': self.get_data_loader(phase='train'),
            'test': self.get_data_loader(phase='test'),
            'observed_train': self.get_observed_data_loaders(phase='train'),
            'observed_test': self.get_observed_data_loaders(phase='test'),
            'task_idx': self.i
        }
        self.data_loaders.update(new_data_loaders)

        self.train_loader = self.data_loaders['train']
        self.test_loader = self.data_loaders['test']

        return self.data_loaders

    def get_backbone(self):
        return AQAMLP(self.args)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss(mse_loss_weight=0.1, plcc_loss_weight=1.0, loss_type='mix'):
        # mse loss
        def mse_loss(output, target):
            return F.mse_loss(output, target)

        # plcc loss
        def plcc_loss(y_pred, y):
            sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
            y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
            sigma, m = torch.std_mean(y, unbiased=False)
            y = (y - m) / (sigma + 1e-8)
            loss0 = F.mse_loss(y_pred, y) / 4
            rho = torch.mean(y_pred * y)
            loss1 = F.mse_loss(rho * y_pred, y) / 4

            return ((loss0 + loss1) / 2).float()

        # mix loss
        def mix_loss(y_pred, y):
            loss0 = mse_loss(y_pred, y)
            loss1 = plcc_loss(y_pred, y)

            if loss_type == 'mse':
                loss = loss0
            elif loss_type == 'plcc':
                loss = loss1
            else:
                loss = loss0 * mse_loss_weight + loss1 * plcc_loss_weight

            return loss

        return mix_loss

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        scheduler = optim.lr_scheduler.StepLR(
            model.opt, step_size=args.decay_interval, gamma=args.decay_ratio)
        return scheduler

    @staticmethod
    def get_batch_size():
        return 128

    @staticmethod
    def get_minibatch_size():
        return 128


if __name__ == '__main__':
    pass
