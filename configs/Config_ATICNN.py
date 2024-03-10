# -*- coding: utf-8 -*-
'''
@time: 2021/4/16 18:45
@ author:
'''

class Config:
    seed = 10

    # path
    datafolder = '../data/ptbxl/'
    resultfolder = './result/'

    experiment = 'exp0'

    model_name = 'ATI_CNN'

    batch_size = 128
    max_epoch = 150
    learning_rate = 0.0001
    patience = 7


config_net = Config()
