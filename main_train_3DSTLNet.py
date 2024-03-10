# -*- coding: utf-8 -*-
"""
@ author: Pan Xia
"""

import torch
from torch import nn, optim
from dataset import load_datasets
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, auc
import random
from tqdm import tqdm
from models.Model import *
from layers.tools import *
from configs.Config_Model import config_mynet
from multilabel_metrics.multilabel_metric import *
from multilabel_metrics.threshold_optimization import *
from multilabel_metrics.Plot_ROC_Confusion_Matrix import *
from data_process import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(epoch, model, optimizer, criterion, train_loader, use_cuda):
    model.train()
    loss_meter, it_count = 0, 0
    inputs = []
    outputs = []
    targets = []
    # train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    # train_bar = tqdm(initial=0, leave=True, total=len(train_loader), desc=train_desc.format(epoch, 0), position=0)
    for input, target in train_loader:
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        input = input + torch.randn_like(input) * 0.1

        # forward
        output = model(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_meter += loss.item()
        it_count += 1

        output = torch.sigmoid(output)
        for i in range(len(output)):
            outputs.append(output[i].cpu().detach().numpy())
            targets.append(target[i].cpu().detach().numpy())
            inputs.append(input[i].cpu().detach().numpy())

    #     train_bar.desc = train_desc.format(epoch, loss_meter / it_count)
    #     train_bar.update(target.size(0))
    #
    # train_bar.close()
    auc = roc_auc_score(targets, outputs)

    print('epoch: %.f, train_loss: %.4f,   macro_auc: %.4f' % (epoch, loss_meter / it_count, auc))
    return loss_meter / it_count, inputs, targets, outputs, auc


# val and test
def test_epoch(epoch, model, criterion, valid_loader, use_cuda):
    model.eval()
    loss_meter, it_count = 0, 0
    outputs = []
    targets = []
    # eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    # eval_bar = tqdm(initial=0, leave=True, total=len(valid_loader), desc=eval_desc.format(epoch, 0), position=0)
    with torch.no_grad():
        for inputs, target in valid_loader:
            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            inputs = inputs + torch.randn_like(inputs) * 0.1

            output = model(inputs)
            loss = criterion(output, target)

            loss_meter += loss.item()
            it_count += 1

            output = torch.sigmoid(output)
            for i in range(len(output)):
                outputs.append(output[i].cpu().detach().numpy())
                targets.append(target[i].cpu().detach().numpy())

        #     eval_bar.desc = eval_desc.format(epoch, loss_meter / it_count)
        #     eval_bar.update(target.size(0))
        # eval_bar.close()

        auc = roc_auc_score(targets, outputs)

    print('epoch: %.f, test_loss: %.4f,   macro_auc: %.4f' % (epoch, loss_meter / it_count, auc))
    return loss_meter / it_count, targets, outputs, auc


def train(config):
    # set seed
    setup_seed(config.seed)

    # 加载数据 10 folds cross-validation
    for fold_index in range(10):
        test_fold = fold_index
        if test_fold == 0:
            val_fold = 9
        else:
            val_fold = test_fold - 1
        train_fold = np.delete(np.arange(10), [val_fold, test_fold])

        train_dataloader, val_dataloader, test_dataloader, num_classes = load_datasets(
            datafolder=config.datafolder, experiment=config.experiment, batch_size=config.batch_size, train_fold=train_fold + 1,
            val_fold=val_fold + 1, test_fold=test_fold + 1)

        # 实例化model
        model = Model(config, num_classes=num_classes, in_channel=1)
        print('model_name:{}, experiment_name:{}, num_classes:{}, fold:{}'.format(config.model_name, config.experiment,
                                                                         num_classes, fold_index+1))
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()

        # optimizer and loss
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        scheduler_lr_net = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50],
                                                                gamma=0.5, last_epoch=-1)

        early_stopping = EarlyStopping(patience=config.patience, verbose=True)

        if not os.path.isdir('result'):
            os.mkdir('result')

        # =========>train<=========
        for epoch in range(1, config.max_epoch + 1):
            train_loss, train_inputs, train_targets, train_outputs, train_auc = train_epoch(epoch, model, optimizer,
                                                                                            criterion,
                                                                                            train_dataloader,
                                                                                            use_cuda)
            val_loss, val_targets, val_outputs, val_auc = test_epoch(epoch, model, criterion, val_dataloader,
                                                                     use_cuda)
            test_loss, test_targets, test_outputs, test_auc = test_epoch(epoch, model, criterion, test_dataloader,
                                                                         use_cuda)

            # calculate metrics
            train_targets = np.array(train_targets)
            train_outputs = np.array(train_outputs)
            val_targets = np.array(val_targets)
            val_outputs = np.array(val_outputs)
            test_targets = np.array(test_targets)
            test_outputs = np.array(test_outputs)

            # OneThreshold
            # threshold_one_optimal = One_threshold(val_targets, val_outputs, num_classes)
            # test_predicts = label_prediction(threshold_one_optimal, test_outputs, num_classes)

            # RCut
            # label_cardinality = calculate_cardinality(val_targets)
            # print(label_cardinality)
            # test_predicts = based_cardinality_predict(test_outputs, num_classes, label_cardinality)

            # SCut hamming loss
            # threshold_class_optimal = specclass_threshold(val_targets, val_outputs, num_classes)
            # test_predicts = label_prediction(threshold_class_optimal, test_outputs, num_classes)

            # ThresholdPrediction TP
            train_threshold(val_targets, val_outputs)
            test_predicts, opti_threshold = threshold_predict(test_outputs)

            subset_accuracy = subsetAccuracy(test_targets, test_predicts)
            hamming_Loss = hammingLoss(test_targets, test_predicts)
            f1_measure = fbeta(test_targets, test_predicts, beta=1)

            print("AUC", test_auc)
            print("subset_accuracy", subset_accuracy)
            print("hamming_Loss", hamming_Loss)
            print("f1_measure", f1_measure)

            early_stopping(-val_auc, model, config.resultfolder)
            scheduler_lr_net.step()


if __name__ == '__main__':
    # ptb-xl tasks
    for exp in ['exp3', 'exp1.1', 'exp1.1.1']:
        if exp == 'exp3':  # 'rhythm'
            config_mynet.seed = 7
        elif exp == 'exp1.1':  # 'subdiagnostic'
            config_mynet.seed = 20
        elif exp == 'exp1.1.1':  # 'superdiagnostic'
            config_mynet.seed = 2023

        config_mynet.experiment = exp
        train(config_mynet)

    # CPSC 2018 task
    config_mynet.datafolder = '../data/CPSC/'
    config_mynet.experiment = 'cpsc'
    config_mynet.seed = 2023
    train(config_mynet)

    # HFHC task
    config_mynet.datafolder = '../data/hf/'
    config_mynet.experiment = 'hf'
    config_mynet.seed = 9
    train(config_mynet)
