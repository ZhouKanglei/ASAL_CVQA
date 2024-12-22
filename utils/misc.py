# -*- coding: utf-8 -*-
# @Time: 2023/6/20 20:37
import argparse
import os
import random
import shutil

import numpy as np
import pickle
import torch
import pandas as pd
import json


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class cmdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + '_non_default', True)


def copy_dir(src, dst):
    if os.path.exists(dst) and os.path.isdir(dst):
        shutil.rmtree(dst)

    if os.path.exists(src):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)


def init_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def mask_classes(outputs, dataset, k):
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def distributed_concat(tensor):
    if torch.distributed.is_initialized() == False:
        return tensor
    output_tensors = [tensor.clone().contiguous()
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor.contiguous())
    concat_tensor = torch.cat(output_tensors, dim=0)
    return concat_tensor


def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    return pickle_data


def logging_info(info, logger=None):
    if logger == 'pass':  # do not log
        return
    if logger is not None:
        logger.info(info)
    else:
        print(info)


def save_score_realtime(pred_scores, label_scores, label_names, save_dir, filename=None, logger=None):
    if filename is None:
        filename = 'results.csv'
    if save_dir is None:
        logging_info("No experiment directory provided!!!", logger)
        return
    # save results into a csv file
    file_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = {'Score': pred_scores,
            'Label': label_scores, 'Name': label_names}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=True)

    logging_info(f"Results saved in {file_path}", logger)


def save_metric_results(results, save_dir, filename=None, logger=None):
    if filename is None:
        filename = 'results.json'
    if save_dir is None:
        logging_info("No experiment directory provided!!!", logger)
        return
    # save results into a json file
    file_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    logging_info(f"Results saved in {file_path}", logger)


def save_all_scores(scores, predicted_scores, names, save_dir, filename=None, logger=None):
    if filename is None:
        filename = 'scores.csv'
    if save_dir is None:
        logging_info("No experiment directory provided!!!", logger)
        return
    # save results into a pickle file
    file_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    data = {'Label Score': scores,
            'Predicted Score': predicted_scores, 'Name': names}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=True)

    logging_info(f"Results saved in {file_path}", logger)