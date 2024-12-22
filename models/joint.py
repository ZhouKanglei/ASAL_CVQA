#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/08/30 15:38:31
import os
import json
import torch

from models.utils.continual_model import ContinualModel

from utils.status import progress_bar
from utils.metrics import stat_results_by_category, print_results
from utils.misc import save_score_realtime, save_metric_results


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)

        self.opt = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        self.best_metric = None

    def train(self, dataset):
        # 1 initialization
        if self.args.model == 'joint_wo_base':
            loader = dataset.data_loaders['joint_train']
            test_loader = dataset.data_loaders['joint_test']
        elif self.args.model == 'joint':
            loader = dataset.data_loaders['joint_train_full']
            test_loader = dataset.data_loaders['joint_test_full']
        else:
            loader = dataset.data_loaders['base_train']
            test_loader = dataset.data_loaders['base_test']

        self.scheduler = dataset.get_scheduler(self, self.args)
        # 2 before training, evaluate the model
        self.evaluate(test_loader, epoch=0)
        # 3 train the model
        for e in range(self.args.n_epochs):
            # judge whether the testing loss is decreasing, else stop training
            if self.args.early_stop and self.best_metric is not None:
                if e - self.best_metric['epoch'] > self.args.patience:
                    self.args.logging.info(
                        'The model has been well trained, early stopping')
                    break
            # initialize the total loss
            total_loss = -1
            label_scores, pred_scores, label_names = [], [], []
            # batch training
            for i, batch in enumerate(loader):
                # for debug
                if i > 5 and self.args.debug_mode:  # debug mode
                    break
                # forward
                inputs, labels, names = batch

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.net(inputs)

                self.opt.zero_grad()

                loss = self.loss(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                self.opt.step()

                if self.args.model == 'base':
                    progress_bar(i, len(loader), e, 'B', loss.item())
                else:
                    progress_bar(i, len(loader), e, 'J', loss.item())

                label_scores.extend(
                    labels.detach().cpu().numpy().reshape(-1,).tolist())
                pred_scores.extend(
                    outputs.detach().cpu().numpy().reshape(-1,).tolist())
                label_names.extend(names)

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()
                print()
                self.args.logging.info(
                    'The current learning rate is {:.06f}'.format(lr[0]))
                self.scheduler.step()
            # print the average loss of the epoch
            avg_loss = total_loss / len(loader)
            self.args.logging.info(f"Average training loss is {avg_loss:.4f}")
            # evaluate the model on the training set
            stat_results_by_category(pred_scores, label_scores, label_names,
                                     info=f'Training epoch: {e + 1}',
                                     logger=self.args.logging)
            save_score_realtime(pred_scores, label_scores, label_names,
                                self.args.output_score_dir, filename=f'train.csv',
                                logger=self.args.logging)
            # evaluate the model on the test set
            results = self.evaluate(test_loader, epoch=e)
            # save best model
            self.save_best_model(results, epoch=e)

        # 4 print the best model information
        print_results(self.best_metric, 'Best model',
                      logger=self.args.logging)
        # 5 save results to json file
        save_metric_results(self.best_metric, self.args.output_result_dir,
                            filename='best_results.json', logger=self.args.logging)

    def evaluate(self, test_loader, epoch=None):
        # set the model to evaluation mode
        status = self.net.training
        self.net.eval()
        # initialize the label and output scores
        label_scores, pred_scores, label_names = [], [], []
        # batch evaluation
        for _, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels, names = data
                inputs, labels = inputs.to(self.args.output_device), \
                    labels.to(self.args.output_device)

                outputs = self.net(inputs)

                label_scores.extend(
                    labels.detach().cpu().numpy().reshape(-1,).tolist())
                pred_scores.extend(
                    outputs.detach().cpu().numpy().reshape(-1,).tolist())
                label_names.extend(names)

        # compute the metrics
        results, self.category_num = stat_results_by_category(
            pred_scores, label_scores, label_names,
            info=f'{epoch + 1}-th evaluation' if epoch != -
            1 else f'Before evaluation',
            logger=self.args.logging)
        save_score_realtime(pred_scores, label_scores, label_names,
                            self.args.output_score_dir, filename=f'test.csv',
                            logger=self.args.logging)
        # set the model back to the original status
        self.net.train(status)

        return results

    def save_best_model(self, metrics, epoch=None):
        if self.best_metric is None:
            self.best_metric = metrics
            self.best_metric['epoch'] = epoch + 1

        if metrics['Overall']['SRCC'] >= self.best_metric['Overall']['SRCC']:
            self.best_metric = metrics
            self.best_metric['epoch'] = epoch + 1
            self.save_model('best_model.pth')
            # log the saving information
            self.args.logging.info(
                f'Best model saved successfully'.center(36, ' ').center(80, '-'))
        else:
            metrics.update({'epoch': epoch + 1})
            self.save_model('last_model.pth', metrics)
        self.args.logging.info(
            'Current best SRCC is {:.4f} ({:d}-th epoch)'.format(
                self.best_metric['Overall']['SRCC'], self.best_metric['epoch']))

    def save_model(self, model_name, metrics=None):
        weight_path = os.path.join(
            self.args.output_model_weight_dir, model_name)
        # save the model
        state_dict = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        state_dict.update(self.best_metric if metrics is None else metrics)

        torch.save(state_dict, weight_path)
