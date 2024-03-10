# -*- coding: utf-8 -*-
'''
@ author: Pan Xia
'''

class Config:
    seed = 10

    # path
    datafolder = '../data/ptbxl/'
    resultfolder = './result/'

    '''
    experiment = exp0, exp1, exp1.1, exp1.1.1, exp2, exp3
    '''
    experiment = 'exp0'

    # Training hyper parameters
    batch_size = 128
    learning_rate = 0.001
    max_epoch = 50
    device_num = 1
    patience = 10
    lradj = "type1"

    model_name = 'TimesNet_II_lead'

    # network hyper parameters
    num_blocks = 2
    in_channel = 1
    d_model = 32
    dropout = 0.3
    top_k = 3
    num_kernels = 3
    d_ff = 32
    seq_len = 125
    pred_len = 0


config_mynet = Config
