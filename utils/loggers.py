# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import suppress
import os
import sys
import json

from typing import Any, Dict

import numpy as np

from utils import create_if_not_exists
from utils.conf import base_path
from utils.metrics import backward_transfer, forward_transfer, forgetting

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
                  mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)


class Logger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        self.accs = []
        self.fullaccs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.fullaccs_mask_classes = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None

    def dump(self):
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
            'fwt_mask_classes': self.fwt_mask_classes,
            'bwt_mask_classes': self.bwt_mask_classes,
            'forgetting_mask_classes': self.forgetting_mask_classes,
        }
        if self.setting == 'class-il':
            dic['accs_mask_classes'] = self.accs_mask_classes
            dic['fullaccs_mask_classes'] = self.fullaccs_mask_classes

        return dic

    def load(self, dic):
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.fwt = dic['fwt']
        self.bwt = dic['bwt']
        self.forgetting = dic['forgetting']
        self.fwt_mask_classes = dic['fwt_mask_classes']
        self.bwt_mask_classes = dic['bwt_mask_classes']
        self.forgetting_mask_classes = dic['forgetting_mask_classes']
        if self.setting == 'class-il':
            self.accs_mask_classes = dic['accs_mask_classes']
            self.fullaccs_mask_classes = dic['fullaccs_mask_classes']

    def rewind(self, num):
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.fwt = self.fwt[:-num]
            self.bwt = self.bwt[:-num]
            self.forgetting = self.forgetting[:-num]
            self.fwt_mask_classes = self.fwt_mask_classes[:-num]
            self.bwt_mask_classes = self.bwt_mask_classes[:-num]
            self.forgetting_mask_classes = self.forgetting_mask_classes[:-num]

        if self.setting == 'class-il':
            self.accs_mask_classes = self.accs_mask_classes[:-num]
            self.fullaccs_mask_classes = self.fullaccs_mask_classes[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(
                results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)

    def log_fullacc(self, accs):
        if self.setting == 'class-il':
            acc_class_il, acc_task_il = accs
            self.fullaccs.append(acc_class_il)
            self.fullaccs_mask_classes.append(acc_task_il)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        wrargs = args.copy()

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

        wrargs['forward_transfer'] = self.fwt
        wrargs['backward_transfer'] = self.bwt
        wrargs['forgetting'] = self.forgetting

        target_folder = base_path() + "results/"

        create_if_not_exists(target_folder + self.setting)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset)
        create_if_not_exists(target_folder + self.setting +
                             "/" + self.dataset + "/" + self.model)

        path = target_folder + self.setting + "/" + self.dataset\
            + "/" + self.model + "/logs.pyd"
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')

        if self.setting == 'class-il':
            create_if_not_exists(os.path.join(
                *[target_folder, "task-il/", self.dataset]))
            create_if_not_exists(target_folder + "task-il/"
                                 + self.dataset + "/" + self.model)

            for i, acc in enumerate(self.accs_mask_classes):
                wrargs['accmean_task' + str(i + 1)] = acc

            for i, fa in enumerate(self.fullaccs_mask_classes):
                for j, acc in enumerate(fa):
                    wrargs['accuracy_' + str(j + 1) +
                           '_task' + str(i + 1)] = acc

            wrargs['forward_transfer'] = self.fwt_mask_classes
            wrargs['backward_transfer'] = self.bwt_mask_classes
            wrargs['forgetting'] = self.forgetting_mask_classes

            path = target_folder + "task-il" + "/" + self.dataset + "/"\
                + self.model + "/logs.pyd"
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')


class Logger_cvqa:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:

        self.task_k_results = None
        self.previous_k_results = None

        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str

        self.fwt = {}
        self.bwt = {}
        self.forgetting = {}

    def dump(self):
        results = {
            'task_k_results': self.task_k_results,
            'previous_k_results': self.previous_k_results,
            'fwt': self.fwt,
            'bwt': self.bwt,
            'forgetting': self.forgetting,
        }
        return results

    def add_fwt(self, results, random_results):
        previous_k_results = results['previous_k_results']
        data = self.parse_dict(previous_k_results)
        random_previous_k_results = random_results['previous_k_results']
        random_data = self.parse_dict_single(random_previous_k_results)
        for k, v in data.items():
            if k not in self.fwt:
                self.fwt[k] = {}
            for k1, v1 in v.items():
                if k1 not in self.fwt[k]:
                    self.fwt[k][k1] = []
                self.fwt[k][k1] = forward_transfer(v1, random_data[k][k1])

    def add_bwt(self, results):

        previous_k_results = results['previous_k_results']
        data = self.parse_dict(previous_k_results)
        for k, v in data.items():
            if k not in self.bwt:
                self.bwt[k] = {}
            for k1, v1 in v.items():
                if k1 not in self.bwt[k]:
                    self.bwt[k][k1] = []
                self.bwt[k][k1] = backward_transfer(v1)

    def add_forgetting(self, results):

        previous_k_results = results['previous_k_results']

        data = self.parse_dict(previous_k_results)
        for k, v in data.items():
            if k not in self.forgetting:
                self.forgetting[k] = {}
            for k1, v1 in v.items():
                if k1 not in self.forgetting[k]:
                    self.forgetting[k][k1] = []
                self.forgetting[k][k1] = forgetting(v1)

    def parse_dict_single(self, data):
        ret = {}

        for k, v in data.items():
            for sub_k, sub_v in v.items():
                if sub_k not in ret:
                    ret.update({sub_k: {}})
                for metric, value in sub_v.items():
                    if metric not in ret[sub_k]:
                        ret[sub_k].update({metric: []})
                    ret[sub_k][metric].append(value)

        return ret

    def parse_dict(self, data):
        ret = {}

        # initialize the dict
        for k, v in data.items():  # task level 1
            for k1, v1 in v.items():  # task level 2
                for k2, v2 in v1.items():  # category level
                    if k2 not in ret:
                        ret[k2] = {}
                    for k3, v3 in v2.items():  # metrics level
                        if k3 not in ret[k2]:
                            ret[k2][k3] = []

        # first concatenate all the results in the task level 2 for each metric
        # then concatenate all the results in the task level 1 for each metric
        for k2, v2 in ret.items():  # category level
            for k3, v3 in v2.items():  # metrics level
                for k, v in data.items():
                    res = []
                    for k1, v1 in v.items():
                        res.append(data[k][k1][k2][k3])
                    ret[k2][k3].append(res)

        return ret

    def log_results(self, results):
        self.previous_k_results = results['previous_k_results']
        self.task_k_results = results['task_k_results']
    
