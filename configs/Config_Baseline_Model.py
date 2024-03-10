# -*- coding: utf-8 -*-

class Config:
    seed = 10

    # path
    datafolder = '../data/ptbxl/'
    resultfolder = './result/'

    experiment = 'exp3'

    # model_name = 'lstm_bidir'
    model_name = 'xresnet1d101'

    batch_size = 64
    max_epoch = 50
    learning_rate = 0.001
    patience = 7


config_net = Config()
