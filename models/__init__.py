# -*- coding: utf-8 -*-
# Time: 2023/6/20 11:54

import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


def get_model(args, backbone, loss, transform):
    if args.model == 'base':
        return names['joint'](backbone, loss, args, transform)
    elif args.model == 'joint_wo_base':
        return names['joint'](backbone, loss, args, transform)
    else:
        return names[args.model](backbone, loss, args, transform)


if __name__ == '__main__':
    print(get_all_models())
