# -*- coding: utf-8 -*-
# @Time: 2023/6/24 20:37
import os
import pprint
import json
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from datasets import get_dataset
from utils.loggers import Logger_cvqa as Logger
from utils.processor import Processor
from utils.status import ProgressBar

from utils.metrics import stat_results_by_category, calculate_average_values, print_results
from utils.misc import save_metric_results, save_all_scores
from utils.loss_landscape import calculate_loss, plot_1d_loss_all


class ProcessorVQA(Processor):

    def __init__(self, args):
        super(ProcessorVQA, self).__init__(args)
        self.args = args

        self.current_epoch = -1
        self.current_task = -1

        self.best_metric = None

        self.best_weight_path = os.path.join(
            os.path.dirname(self.args.output_model_weight_dir), 'best_model.pth')
        self.last_weight_path = os.path.join(
            os.path.dirname(self.args.output_model_weight_dir), 'last_model.pth')

    def save_checkpoint(self, results, filename='last'):

        checkpoint_file = os.path.join(
            self.args.output_model_weight_dir, filename)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.net.state_dict(),
            'task': self.current_task,
            'results': results
        }, checkpoint_file)

        self.args.logging.info(f'Save checkpoint to {checkpoint_file}.')

    def load_checkpoint(self, checkpoint_file):

        if not os.path.exists(checkpoint_file):
            self.args.logging.info(
                f'Checkpoint file {checkpoint_file} does not exist.')
            return

        checkpoint = torch.load(checkpoint_file)
        self.model.net.load_state_dict(
            checkpoint['model_state_dict'], strict=False)
        self.args.logging.info(f'Load checkpoint from {checkpoint_file}.')

    def evaluate(self, dataset, last=False):
        # set the model to evaluation mode
        status = self.model.net.training
        self.model.net.eval()

        # initialize the label and output scores
        self.task_k_results = {}
        self.previous_k_results = {}
        all_label_scores, all_output_scores = [], []
        all_label_names = []

        observed_test_loaders = dataset.data_loaders['observed_test']
        # task-wise evaluation
        for k, test_loader in enumerate(observed_test_loaders):
            # only evaluate the last task
            if last and k < len(observed_test_loaders) - 1:
                continue
            # phase == test, load the current task's model
            if self.args.phase == 'test' and k == 0:
                self.load_checkpoint(
                    f'{self.args.output_model_weight_dir}/last_model-task{dataset.i + 1}.pth')

                if hasattr(self.model, 'save_buffer_feats'):
                    self.model.save_buffer_feats(dataset)

                if hasattr(self.model, 'save_test_feats'):
                    self.model.save_test_feats(dataset)

            label_scores, output_scores = [], []
            label_names = []
            # batch evaluation
            for data in test_loader:
                with torch.no_grad():
                    inputs, labels, names = data
                    inputs, labels = inputs.to(self.args.output_device), \
                        labels.to(self.args.output_device)

                    outputs = self.model(inputs)

                    label_scores.extend(
                        labels.detach().cpu().numpy().reshape(-1,).tolist())
                    output_scores.extend(
                        outputs.detach().cpu().numpy().reshape(-1,).tolist())
                    label_names.extend(names)

            # compute the metrics of the k-th task
            task_k_result, self.task_k_category_num = stat_results_by_category(
                output_scores, label_scores, label_names,
                info=f'{k + 1}-th task evaluation',
                logger='pass')
            self.task_k_results.update({k + 1: task_k_result})

            # log the current task k and the previous tasks
            all_label_scores.extend(label_scores)
            all_output_scores.extend(output_scores)
            all_label_names.extend(label_names)

            # log the overall metrics from the second task
            previous_k_result, self.previous_k_category_num = stat_results_by_category(
                all_output_scores, all_label_scores, all_label_names,
                info=f'Previous {k + 1} evaluation',
                logger='pass')
            self.previous_k_results.update({k + 1: previous_k_result})

        # task-wise evaluation
        calculate_average_values(self.task_k_results, info='Average evaluation',
                                 logger=self.args.logging, category_num=self.task_k_category_num)
        print_results(self.previous_k_results[k + 1], info='Overall evaluation',
                      logger=self.args.logging, category_num=self.previous_k_category_num)

        # save the results
        save_metric_results(self.task_k_results, self.args.output_result_dir,
                            filename=f'task_k_results-{self.args.phase}.json',
                            logger=self.args.logging)
        save_metric_results(self.previous_k_results, self.args.output_result_dir,
                            filename=f'previous_k_results-{self.args.phase}.json',
                            logger=self.args.logging)

        # calculate the disturbuance loss of the model at the end of all tasks
        if self.args.phase == 'test' and k == dataset.N_TASKS - 1:
            if self.args.visual_landscape:
                self.steps = np.arange(
                    self.args.step_min, self.args.step_max, self.args.step_size)
                self.loss_landscape = calculate_loss(
                    self.model, self.dataset, self.steps,
                    dir_num=self.args.dir_num, output_dir=self.args.output_dir)

        # save labels and predicted scores
        if self.args.phase == 'test':
            save_all_scores(all_label_scores, all_output_scores, all_label_names,
                            self.args.output_score_dir,
                            filename=f'all_scores-task{k+1}.csv', logger=self.args.logging)

        # set the model back to the original status
        self.model.net.train(status)

        return self.task_k_results, self.previous_k_results

    def training_loop(self, train_loader, progress_bar, scheduler, t):
        # sequential learning
        for epoch in range(self.args.n_epochs):

            self.current_epoch = epoch
            total_loss = 0
            # batch training
            for i, data in enumerate(train_loader):

                # debug mode
                if self.args.debug_mode and i > 3:
                    break

                inputs, labels, names = data
                not_aug_inputs = inputs.clone()
                inputs, labels = inputs.to(self.args.output_device), \
                    labels.to(self.args.output_device)
                not_aug_inputs = not_aug_inputs.to(self.args.output_device)
                loss = self.model.meta_observe(
                    inputs, labels, not_aug_inputs, epoch, t)

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

                total_loss += loss

            # print the average loss of the epoch
            avg_loss = total_loss / len(train_loader)
            # self.args.logging.info(f"Average training loss is {avg_loss:.4f}")

            # scheduler
            if scheduler is not None:
                scheduler.step()

        # new line, split the training bar
        print()

    def train(self):
        # 1 random model for calculating fwt for non-joint/continual models
        dataset_copy = get_dataset(self.args)
        self.dataset_copy = dataset_copy
        # 1.1 jumpy to the last session (i == N_TASKS - 1)
        for t in range(self.dataset.N_TASKS):
            self.model.net.train()
            dataset_copy.get_data_loaders()
        # 1.2 evaluate from the last session
        self.random_task_k_results, self.random_previous_k_results \
            = self.evaluate(dataset_copy)
        self.random_results = {
            'task_k_results': self.random_task_k_results,
            'previous_k_results': self.random_previous_k_results,
        }

        # 2 training model
        self.results = {}
        logger = Logger(self.dataset.SETTING,
                        self.dataset.NAME, self.model.NAME)
        progress_bar = ProgressBar(verbose=True)

        # 2.1 load the base model
        if self.args.base_pretrain:
            self.load_checkpoint(self.args.base_pretrain_model_path)

        for t in range(self.dataset.N_TASKS):
            # 2.2 initializing
            self.current_task = t + 1
            self.args.logging.info(f'| Task {t + 1:02d} |'.center(80, '-'))

            self.model.net.train()
            loaders = self.dataset.get_data_loaders()
            train_loader = loaders['train']  # current task train loader
            scheduler = self.dataset.get_scheduler(self.model, self.args)

            # 2.3 forward the future task for the last session
            if t:
                self.evaluate(self.dataset)
                result = self.results['task_k_results'][t]
                result.update({t + 1: self.task_k_results[t + 1]})
                self.results['task_k_results'].update({t: result})

                result = self.results['previous_k_results'][t]
                result.update({t + 1: self.previous_k_results[t + 1]})
                self.results['previous_k_results'].update({t: result})

            # 2.4 beginning task
            if hasattr(self.model, 'begin_task'):
                self.model.begin_task(self.dataset)

            # 2.5 middle task
            self.training_loop(train_loader, progress_bar, scheduler, t)

            # 2.6 ending task
            if hasattr(self.model, 'end_task'):
                self.model.end_task(self.dataset)

            # 2.7 evaluation
            self.evaluate(self.dataset)

            # 2.8 task level statistics
            self.task_stat(logger)

        # 3 final statistics
        self.final_stat(logger)

    def task_stat(self, logger):
        # update the results
        if 'task_k_results' not in self.results:
            self.results.update({"task_k_results": {}})
        if 'previous_k_results' not in self.results:
            self.results.update({"previous_k_results": {}})
        self.results['task_k_results'].update(
            {self.current_task: self.task_k_results})
        self.results['previous_k_results'].update(
            {self.current_task: self.previous_k_results})
        logger.log_results(self.results)

        # save the model
        self.save_checkpoint(self.results,
                             filename=f'last_model-task{self.current_task}.pth')

    def final_stat(self, logger):
        # metric
        logger.add_bwt(self.results)
        logger.add_forgetting(self.results)
        logger.add_fwt(self.results, self.random_results)
        # log
        logger.log_results(self.results)
        results = logger.dump()
        print_results(results['forgetting'],
                      info='Final results - forgetting', logger=self.args.logging,
                      category_num=self.previous_k_category_num)
        print_results(results['fwt'],
                      info='Final results - fwt', logger=self.args.logging,
                      category_num=self.previous_k_category_num)
        print_results(results['bwt'],
                      info='Final results - bwt', logger=self.args.logging,
                      category_num=self.previous_k_category_num)
        # save results
        save_metric_results(results, self.args.output_result_dir,
                            filename='final_results.json', logger=self.args.logging)

    def test(self):
        dataset = self.dataset
        # 1.1 jumpy to the last session (i == N_TASKS - 1)
        for _ in range(self.dataset.N_TASKS):
            dataset.get_data_loaders()
            # 1.2 evaluate the model
            self.evaluate(dataset)

    def save_vis_feats(self, dataset, last=False):
        pass

    def start(self):
        # train and test
        if self.args.phase == 'train':
            # training
            if 'base' in self.args.model:
                self.model.train(self.dataset)  # base session training
                # generalization evaluation
                for _ in range(self.dataset.N_TASKS):
                    self.dataset.get_data_loaders()
                self.evaluate(self.dataset)

            elif 'joint' in self.args.model:
                self.model.train(self.dataset)  # joint training
            else:
                self.train()  # sequential training

            self.args.logging.info('Training finished.')
        else:
            # testing
            self.test()
            self.args.logging.info('Testing finished.')
