# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata
from scipy.optimize import curve_fit
from scipy import stats

from utils.misc import logging_info


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + \
        np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_output, y_label):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output,
                        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)

    return y_output_logistic


def compute_metrics(pred_scores, true_scores):
    # corelation
    try:
        PLCC = stats.pearsonr(pred_scores, true_scores)[0]
    except ValueError:
        PLCC = np.nan
    # PLCC = stats.pearsonr(pred_scores, true_scores)[0]
    SRCC = stats.spearmanr(pred_scores, true_scores)[0]

    # absolute error
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = 100 * np.power((pred_scores - true_scores) / (true_scores.max() -
                         true_scores.min()), 2).sum() / true_scores.shape[0]

    results = {
        'SRCC': SRCC,
        'PLCC': PLCC,
        'L2': L2,
        'RL2': RL2
    }

    return results


def stat_results_by_category(pred_scores, label_scores, label_names, info=None, logger=None):
    if info is not None:
        logging_info(info.center(len(info) + 6, ' ').center(80, '-'), logger)
    else:
        logging_info('-' * 80, logger)
    # compute the metrics for each category
    category_metrics = {}
    category_num = {}
    for category in ['7_A', '7_B', '7', '15_A', '15_B', '15']:
        category_scores = []
        for output_score, label_score, label_name in zip(pred_scores, label_scores, label_names):
            if label_name.startswith(category):
                category_scores.append((output_score, label_score))
        if len(category_scores) == 0:
            continue
        metrics = compute_metrics(*zip(*category_scores))
        category_metrics[category] = metrics
        category_num[category] = len(category_scores)
        logging_info(f'{category:>8s}, {len(category_scores):4d} samples, '
                     f'SRCC: {metrics["SRCC"]:7.4f}, PLCC: {metrics["PLCC"]:7.4f}, '
                     f'L2: {metrics["L2"]:7.4f}, RL2: {metrics["RL2"]:7.4f}', logger)

    # compute the overall metrics
    category_metrics['Overall'] = compute_metrics(
        pred_scores, label_scores)
    # log the overall metrics
    category = 'Overall'
    metrics = category_metrics[category]
    category_num[category] = len(pred_scores)
    logging_info(f'{category:>8s}, {len(pred_scores):4d} samples, '
                 f'SRCC: {metrics["SRCC"]:7.4f}, PLCC: {metrics["PLCC"]:7.4f}, '
                 f'L2: {metrics["L2"]:7.4f}, RL2: {metrics["RL2"]:7.4f}',
                 logger)
    logging_info('-' * 80, logger)

    return category_metrics, category_num


def print_results(results, info, logger=None, category_num=None):
    if info is not None:
        if 'epoch' in results:
            info = info + f' (epoch {results["epoch"]})'
        logging_info(info.center(len(info) + 6, ' ').center(80, '-'), logger)
    else:
        logging_info('-' * 80, logger)

    for category, metrics in results.items():
        if category == 'epoch':
            continue
        if category_num is not None:
            logging_info(f'{category:>8s}, {category_num[category]:4d} samples, '
                         f'SRCC: {metrics["SRCC"]:7.4f}, PLCC: {metrics["PLCC"]:7.4f}, '
                         f'L2: {metrics["L2"]:7.4f}, RL2: {metrics["RL2"]:7.4f}',
                         logger)
        else:
            logging_info(f'{category:>14s}, '
                         f'SRCC: {metrics["SRCC"]:7.4f}, PLCC: {metrics["PLCC"]:7.4f}, '
                         f'L2: {metrics["L2"]:7.4f}, RL2: {metrics["RL2"]:7.4f}', logger)
    logging_info('-' * 80, logger)


def calculate_average_values(data, info='Average', logger=None, category_num=None):
    mean_result = {}
    number = len(data)
    for k, v in data.items():
        for sub_k, sub_v in v.items():
            if sub_k not in mean_result:
                mean_result.update({sub_k: {}})
            for metric, value in sub_v.items():
                if metric not in mean_result[sub_k]:
                    mean_result[sub_k].update({metric: 0})
                mean_result[sub_k][metric] += value / number

    print_results(mean_result, info, logger, category_num=category_num)

    return mean_result


class ListNetLoss(nn.Module):
    def __init__(self, eps=1e-7, padded_value_indicator=-1):
        super(ListNetLoss, self).__init__()

        self.eps = eps
        self.padded_value_indicator = padded_value_indicator

    def minmax(self, inputs):
        min_value, _ = inputs.min(dim=-1, keepdim=True)
        max_value, _ = inputs.max(dim=-1, keepdim=True)

        return (inputs - min_value + self.eps/2) / (max_value - min_value + self.eps)

    def listNet(self, y_pred, y_true, kl=True):
        """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        pred_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        # pred_smax = self.minmax(y_pred)
        # true_smax = self.minmax(y_true)

        if kl:
            jsd = 0.5 * self.kld(pred_smax, true_smax) + \
                0.5 * self.kld(true_smax, pred_smax)
            return jsd

        pred_smax = pred_smax + self.eps
        pred_log = torch.log(pred_smax)

        return torch.mean(-torch.sum(true_smax * pred_log, dim=1))

    def kld(self, p, q):
        # p : batch x n_items
        # q : batch x n_items
        return (p * torch.log2(p / q + self.eps)).sum()

    def euclidean_dist(self, inputs):
        batch_size = inputs.shape[0]
        # Euclidean distance
        dist = torch.pow(inputs, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        return dist

    def angular_dist(self, inputs):
        # Angle distance
        # normalize the vectors
        n_inputs = inputs / (torch.norm(inputs, dim=-1, keepdim=True))
        # compute cosine of the angles using dot product
        cos_ij = torch.einsum('bm, cm->bc', n_inputs, n_inputs)
        cos_ij = cos_ij.clamp(min=-1 + self.eps, max=1 - self.eps)
        ang_dist = torch.acos(cos_ij)

        return ang_dist

    def forward(self, inputs_, targets_, blocking=None, wo_iig=False, wo_jg=False):
        # transformation
        if len(inputs_.shape) == 3:
            inputs = inputs_.mean(-1)
        else:
            inputs = inputs_

        inputs = inputs_.reshape(inputs_.shape[0], -1)
        targets = targets_.reshape(targets_.shape[0], -1)

        # prediction
        dist = self.angular_dist(inputs)
        # gt
        diff = torch.abs(targets - targets.t())
        diff_rank = diff.argsort(descending=False).argsort(
            descending=False).to(torch.float32)

        # block
        if blocking == None:
            loss = self.listNet(dist, diff)
        else:
            b1 = blocking
            loss_joint = self.listNet(dist, diff)

            loss11 = self.listNet(dist[:b1, :b1], diff[:b1, :b1])
            loss12 = self.listNet(dist[:b1, b1:], diff[:b1, b1:])
            loss21 = self.listNet(dist[b1:, :b1], diff[b1:, :b1])
            loss22 = self.listNet(dist[b1:, b1:], diff[b1:, b1:])

            loss = loss_joint + loss11 + loss12 + loss21 + loss22

            if wo_iig:
                loss = loss_joint
            if wo_jg:
                loss = loss11 + loss12 + loss21 + loss22

        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        
        inputs = inputs.view(n, -1)
        targets = targets.view(n, -1)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.matmul(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight
