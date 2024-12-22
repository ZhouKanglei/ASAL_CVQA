#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2024/08/30 15:40:44

from utils.processor_vqa import ProcessorVQA as Processor
from utils.parser import Parser
import torch.multiprocessing
import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    args = Parser().args

    processor = Processor(args)

    processor.start()
