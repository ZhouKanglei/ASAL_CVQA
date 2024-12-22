#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time: 2024/05/31 23:24:39

import numpy as np
import pandas as pd
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

plt.rcParams['font.size'] = 18
plt.rcParams['lines.markersize'] = 12

def parse_data(data):
    res = {}
    regex = r'Evaluate T.*\n'
    results = re.findall(regex, data)
    for result in results:
        result = result.strip().split(',')[2:]
        for r in result:
            r = r.split(':')
            key = r[0].strip()
            value = float(r[1].replace('%', '').strip())
            if key not in res.keys():
                res[key] = []
            res[key].append(value)
    
    return res
    

def parse_log_file(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # find the current task number from the content
    regex = r'Rho \(overall\): \d+\.\d+'
    data = {}
    results = re.findall(regex, content)
    for result in results:
        idx = content.find(result)
        c = content[idx-1170:idx]
        # parse the data
        res = parse_data(c)
        for key in res.keys():
            if key not in data.keys():
                data[key] = []
            data[key].extend(res[key])
    # reshape the data
    for key in data.keys():
        data[key] = np.array(data[key]).reshape(-1, len(results))

    return data


def vis_heatmap(matrx, title, save_path=None, mask=None):
    # use seaborn to vis the heatmap, the upper triangle matrix and the lower triangle matrix use different color bars
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # numpy to pd.DataFrame
    matrx = pd.DataFrame(matrx, 
                         columns=range(1, matrx.shape[1]+1),
                         index=range(1, matrx.shape[1]+1))
    sns.heatmap(matrx, cmap="YlGnBu", annot=True, fmt=".2f", 
                square=True, linewidth=.5, cbar_kws={"shrink": 0.8},
                mask=mask)

    # Set title
    ax.set_title(title)

    # Save the figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Save the figure to {save_path}")


if __name__ == "__main__":
    log_file = 'logs/train_logs-mtl-0601_0534.log'
    data = parse_log_file(log_file)
    for key in data.keys():
        fig_file_path = f'outputs/figs/zero_shot/{key}.pdf'
        vis_heatmap(data[key][1:,1:], key, save_path=fig_file_path)

    log_file = '/home/zhangxingxing/Codes/MAGR+/logs/train_logs-mtl_wild_gr-0601_1852.log'
    # log_file = 'outputs/zhangxingxing-fscl/class-mtl/debug/logs/train-20240601-195748.log'
    data = parse_log_file(log_file)
    for key in data.keys():
        fig_file_path = f'outputs/figs/zero_shot/wild_fea/{key}.pdf'
        m = data[key][1:,1:]
        mask = np.zeros_like(m)
        mask[:,-1] = True
        mask[-1,:] = True
        vis_heatmap(m, key, save_path=fig_file_path, mask=None)