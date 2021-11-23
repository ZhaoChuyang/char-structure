#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Loss function for Image Classification.


import torch.nn as nn

from src.modeling.utils.mse_loss import MseLoss


BASE_LOSS_DICT = dict(
    mse_loss=0,
)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.func_list = [MseLoss(reduction='sum'),]

    def forward(self, out_list):
        loss_dict = out_list[-1]
        out_dict = dict()
        weight_dict = dict()
        for key, item in loss_dict.items():
            # print(key, item)
            out_dict[key] = self.func_list[int(item['type'].float().mean().item())](*item['params'])
            weight_dict[key] = item['weight'].mean().item()

        loss = 0.0

        for key in out_dict:
            loss += out_dict[key] * weight_dict[key]

        out_dict['loss'] = loss
        return out_dict
