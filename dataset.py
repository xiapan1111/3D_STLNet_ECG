import numpy as np, os
import torch
from torch.utils.data import DataLoader, Dataset
from data_process import load_dataset, compute_label_aggregations, select_data, preprocess_signals, data_slice, hf_dataset
import matplotlib.pyplot as plt

class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        x = x.transpose()

        x = torch.tensor(x.copy(), dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x, y

    def __len__(self):
        return len(self.data)

# class DownLoadECGData:
#     '''
#         All experiments data
#     '''
#
#     def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
#                  train_fold=8, val_fold=9, test_fold=10):
#         self.min_samples = min_samples
#         self.task = task
#         self.train_fold = train_fold
#         self.val_fold = val_fold
#         self.test_fold = test_fold
#         self.experiment_name = experiment_name
#         self.datafolder = datafolder
#         self.sampling_frequency = sampling_frequency
#
#     def preprocess_data(self):
#         # Load PTB-XL data
#         data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
#         # Preprocess label data
#         labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)
#
#         # Select relevant data and convert to one-hot
#         data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)
#         # CPSC one-hot ranking IAVB, AF, CLBBB, CRBBB, Norm, PAC, STD, STE, PVC
#         # rhythm one-hot ranking AFIB, AFLT, BIGU, PACE, PSVT, SARRH, SBRAD, SR, STACH, SVARR, SVTAC, TRIGU
#         # print(labels)
#         # print(Y)
#         # for j in range(9):
#         #     for i in range(len(labels)):
#         #         if Y[i][j] == 1:
#         #             print(np.asarray(labels)[i])
#         #             print(Y[i])
#
#         # figure_name_norm = os.path.join('result', 'Norm' + '_ECG.png')
#         # for i in range(len(labels)):
#         #     if Y[i][4] == 1 and i == 15:
#         #         print('index: ', i)
#         #         Norm_data = data[i]
#         #         plt.figure(1)
#         #         plt.subplot(611)
#         #         plt.plot(Norm_data[:,0][150:400], linewidth=3)
#         #         plt.subplot(612)
#         #         plt.plot(Norm_data[:,1][150:400], linewidth=3)
#         #         plt.subplot(613)
#         #         plt.plot(Norm_data[:,6][150:400], linewidth=3)
#         #         plt.subplot(614)
#         #         plt.plot(Norm_data[:,7][150:400], linewidth=3)
#         #         plt.subplot(615)
#         #         plt.plot(Norm_data[:,10][150:400], linewidth=3)
#         #         plt.subplot(616)
#         #         plt.plot(Norm_data[:,11][150:400], linewidth=3)
#         #         plt.show()
#         #         plt.savefig(figure_name_norm, dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(2)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,0][150:400], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,1][150:400], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,6][150:400], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'Norm' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(3)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,7][150:400], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,10][150:400], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,11][150:400], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'Norm' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #         break
#
#         # figure_name_pvc = os.path.join('result', 'PVC' + '_ECG.png')
#         # for i in range(len(labels)):
#         #     if Y[i][8] == 1 and i == 328:
#         #         print('index: ', i)
#         #         Norm_data = data[i]
#         #         plt.figure(1)
#         #         plt.subplot(611)
#         #         plt.plot(Norm_data[:,0][400:650], linewidth=3)
#         #         plt.subplot(612)
#         #         plt.plot(Norm_data[:,1][400:650], linewidth=3)
#         #         plt.subplot(613)
#         #         plt.plot(Norm_data[:,6][400:650], linewidth=3)
#         #         plt.subplot(614)
#         #         plt.plot(Norm_data[:,7][400:650], linewidth=3)
#         #         plt.subplot(615)
#         #         plt.plot(Norm_data[:,10][400:650], linewidth=3)
#         #         plt.subplot(616)
#         #         plt.plot(Norm_data[:,11][400:650], linewidth=3)
#         #         plt.show()
#         #         plt.savefig(figure_name_pvc, dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(2)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,0][400:650], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,1][400:650], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,6][400:650], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'PVC' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(3)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,7][400:650], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,10][400:650], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,11][400:650], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'PVC' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #         break
#
#         # figure_name_LBBB = os.path.join('result', 'LBBB' + '_ECG.png')
#         # for i in range(len(labels)):
#         #     if Y[i][2] == 1 and i == 397:
#         #         print('index: ', i)
#         #         Norm_data = data[i]
#         #         plt.figure(1)
#         #         plt.subplot(611)
#         #         plt.plot(Norm_data[:,0][750:1000], linewidth=3)
#         #         plt.subplot(612)
#         #         plt.plot(Norm_data[:,1][750:1000], linewidth=3)
#         #         plt.subplot(613)
#         #         plt.plot(Norm_data[:,6][750:1000], linewidth=3)
#         #         plt.subplot(614)
#         #         plt.plot(Norm_data[:,7][750:1000], linewidth=3)
#         #         plt.subplot(615)
#         #         plt.plot(Norm_data[:,10][750:1000], linewidth=3)
#         #         plt.subplot(616)
#         #         plt.plot(Norm_data[:,11][750:1000], linewidth=3)
#         #         plt.show()
#         #         plt.savefig(figure_name_LBBB, dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(2)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,0][750:1000], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,1][750:1000], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,6][750:1000], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'LBBB' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(3)
#         #         plt.subplot(311)
#         #         plt.plot(Norm_data[:,7][750:1000], linewidth=4)
#         #         plt.subplot(312)
#         #         plt.plot(Norm_data[:,10][750:1000], linewidth=4)
#         #         plt.subplot(313)
#         #         plt.plot(Norm_data[:,11][750:1000], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'LBBB' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
#         #         plt.close()
#         #         break
#
#         # figure_name_norm = os.path.join('result', 'Abstract' + '_ECG.png')
#         # for i in range(len(labels)):
#         #     if sum(Y[i]) >= 2:
#         #         print('index: ', i)
#         #         print(Y[i])
#         #         Norm_data = data[i]
#         #         plt.figure(1)
#         #         plt.subplot(611)
#         #         plt.plot(Norm_data[:, 0][150:600], linewidth=4)
#         #         plt.subplot(612)
#         #         plt.plot(Norm_data[:, 1][150:600], linewidth=4)
#         #         plt.subplot(613)
#         #         plt.plot(Norm_data[:, 2][150:600], linewidth=4)
#         #         plt.subplot(614)
#         #         plt.plot(Norm_data[:, 3][150:600], linewidth=4)
#         #         plt.subplot(615)
#         #         plt.plot(Norm_data[:, 4][150:600], linewidth=4)
#         #         plt.subplot(616)
#         #         plt.plot(Norm_data[:, 5][150:600], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'Abstract' + '_ECG_I_II_III_aVR_aVF_aVL.svg'), dpi=600,
#         #                     bbox_inches='tight')
#         #         plt.close()
#         #
#         #         plt.figure(2)
#         #         plt.subplot(611)
#         #         plt.plot(Norm_data[:, 6][150:600], linewidth=4)
#         #         plt.subplot(612)
#         #         plt.plot(Norm_data[:, 7][150:600], linewidth=4)
#         #         plt.subplot(613)
#         #         plt.plot(Norm_data[:, 8][150:600], linewidth=4)
#         #         plt.subplot(614)
#         #         plt.plot(Norm_data[:, 9][150:600], linewidth=4)
#         #         plt.subplot(615)
#         #         plt.plot(Norm_data[:, 10][150:600], linewidth=4)
#         #         plt.subplot(616)
#         #         plt.plot(Norm_data[:, 11][150:600], linewidth=4)
#         #         plt.show()
#         #         plt.savefig(os.path.join('result', 'Abstract' + '_ECG_V1_V2_V3_V4_V5_V6.svg'), dpi=600,
#         #                     bbox_inches='tight')
#         #         plt.close()
#         #         input()
#
#         # Y_ = np.asarray(Y)
#         # Y_sum = np.sum(Y_, axis=0)
#         # print(Y_sum)
#
#         if self.datafolder == '../data/CPSC/':
#             data = data_slice(data)
#
#         # 10th fold for testing (9th for now)
#         X_test = data[labels.strat_fold == self.test_fold]
#         y_test = Y[labels.strat_fold == self.test_fold]
#         # 9th fold for validation (8th for now)
#         X_val = data[labels.strat_fold == self.val_fold]
#         y_val = Y[labels.strat_fold == self.val_fold]
#         # rest for training
#         X_train = data[labels.strat_fold <= self.train_fold]
#         y_train = Y[labels.strat_fold <= self.train_fold]
#
#         # Preprocess signal data
#         X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)
#
#         return X_train, y_train, X_val, y_val, X_test, y_test

def kth_fold_data(experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                   train_fold=8, val_fold=9, test_fold=10):
    # Load PTB-XL data
    data, raw_labels = load_dataset(datafolder, sampling_frequency)
    # print(data.shape)
    # print(raw_labels.shape)
    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, datafolder, task)
    # print(labels)
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(data, labels, task, min_samples)
    # print(data.shape)
    # print(Y.sum(axis=0))
    # input()

    # CPSC one-hot ranking IAVB, AF, CLBBB, CRBBB, Norm, PAC, STD, STE, PVC
    # rhythm one-hot ranking AFIB, AFLT, BIGU, PACE, PSVT, SARRH, SBRAD, SR, STACH, SVARR, SVTAC, TRIGU
    # print(labels)
    # print(Y)
    # for j in range(9):
    #     for i in range(len(labels)):
    #         if Y[i][j] == 1:
    #             print(np.asarray(labels)[i])
    #             print(Y[i])

    # figure_name_norm = os.path.join('result', 'Norm' + '_ECG.png')
    # for i in range(len(labels)):
    #     if Y[i][4] == 1 and i == 15:
    #         print('index: ', i)
    #         Norm_data = data[i]
    #         plt.figure(1)
    #         plt.subplot(611)
    #         plt.plot(Norm_data[:,0][150:400], linewidth=3)
    #         plt.subplot(612)
    #         plt.plot(Norm_data[:,1][150:400], linewidth=3)
    #         plt.subplot(613)
    #         plt.plot(Norm_data[:,6][150:400], linewidth=3)
    #         plt.subplot(614)
    #         plt.plot(Norm_data[:,7][150:400], linewidth=3)
    #         plt.subplot(615)
    #         plt.plot(Norm_data[:,10][150:400], linewidth=3)
    #         plt.subplot(616)
    #         plt.plot(Norm_data[:,11][150:400], linewidth=3)
    #         plt.show()
    #         plt.savefig(figure_name_norm, dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(2)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,0][150:400], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,1][150:400], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,6][150:400], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'Norm' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(3)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,7][150:400], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,10][150:400], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,11][150:400], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'Norm' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #         break

    # figure_name_pvc = os.path.join('result', 'PVC' + '_ECG.png')
    # for i in range(len(labels)):
    #     if Y[i][8] == 1 and i == 328:
    #         print('index: ', i)
    #         Norm_data = data[i]
    #         plt.figure(1)
    #         plt.subplot(611)
    #         plt.plot(Norm_data[:,0][400:650], linewidth=3)
    #         plt.subplot(612)
    #         plt.plot(Norm_data[:,1][400:650], linewidth=3)
    #         plt.subplot(613)
    #         plt.plot(Norm_data[:,6][400:650], linewidth=3)
    #         plt.subplot(614)
    #         plt.plot(Norm_data[:,7][400:650], linewidth=3)
    #         plt.subplot(615)
    #         plt.plot(Norm_data[:,10][400:650], linewidth=3)
    #         plt.subplot(616)
    #         plt.plot(Norm_data[:,11][400:650], linewidth=3)
    #         plt.show()
    #         plt.savefig(figure_name_pvc, dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(2)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,0][400:650], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,1][400:650], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,6][400:650], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'PVC' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(3)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,7][400:650], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,10][400:650], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,11][400:650], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'PVC' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #         break

    # figure_name_LBBB = os.path.join('result', 'LBBB' + '_ECG.png')
    # for i in range(len(labels)):
    #     if Y[i][2] == 1 and i == 397:
    #         print('index: ', i)
    #         Norm_data = data[i]
    #         plt.figure(1)
    #         plt.subplot(611)
    #         plt.plot(Norm_data[:,0][750:1000], linewidth=3)
    #         plt.subplot(612)
    #         plt.plot(Norm_data[:,1][750:1000], linewidth=3)
    #         plt.subplot(613)
    #         plt.plot(Norm_data[:,6][750:1000], linewidth=3)
    #         plt.subplot(614)
    #         plt.plot(Norm_data[:,7][750:1000], linewidth=3)
    #         plt.subplot(615)
    #         plt.plot(Norm_data[:,10][750:1000], linewidth=3)
    #         plt.subplot(616)
    #         plt.plot(Norm_data[:,11][750:1000], linewidth=3)
    #         plt.show()
    #         plt.savefig(figure_name_LBBB, dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(2)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,0][750:1000], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,1][750:1000], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,6][750:1000], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'LBBB' + '_ECG_I_II_V1.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(3)
    #         plt.subplot(311)
    #         plt.plot(Norm_data[:,7][750:1000], linewidth=4)
    #         plt.subplot(312)
    #         plt.plot(Norm_data[:,10][750:1000], linewidth=4)
    #         plt.subplot(313)
    #         plt.plot(Norm_data[:,11][750:1000], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'LBBB' + '_ECG_V2_V5_V6.png'), dpi=600, bbox_inches='tight')
    #         plt.close()
    #         break

    # figure_name_norm = os.path.join('result', 'Abstract' + '_ECG.png')
    # for i in range(len(labels)):
    #     if sum(Y[i]) >= 2:
    #         print('index: ', i)
    #         print(Y[i])
    #         Norm_data = data[i]
    #         plt.figure(1)
    #         plt.subplot(611)
    #         plt.plot(Norm_data[:, 0][150:600], linewidth=4)
    #         plt.subplot(612)
    #         plt.plot(Norm_data[:, 1][150:600], linewidth=4)
    #         plt.subplot(613)
    #         plt.plot(Norm_data[:, 2][150:600], linewidth=4)
    #         plt.subplot(614)
    #         plt.plot(Norm_data[:, 3][150:600], linewidth=4)
    #         plt.subplot(615)
    #         plt.plot(Norm_data[:, 4][150:600], linewidth=4)
    #         plt.subplot(616)
    #         plt.plot(Norm_data[:, 5][150:600], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'Abstract' + '_ECG_I_II_III_aVR_aVF_aVL.svg'), dpi=600,
    #                     bbox_inches='tight')
    #         plt.close()
    #
    #         plt.figure(2)
    #         plt.subplot(611)
    #         plt.plot(Norm_data[:, 6][150:600], linewidth=4)
    #         plt.subplot(612)
    #         plt.plot(Norm_data[:, 7][150:600], linewidth=4)
    #         plt.subplot(613)
    #         plt.plot(Norm_data[:, 8][150:600], linewidth=4)
    #         plt.subplot(614)
    #         plt.plot(Norm_data[:, 9][150:600], linewidth=4)
    #         plt.subplot(615)
    #         plt.plot(Norm_data[:, 10][150:600], linewidth=4)
    #         plt.subplot(616)
    #         plt.plot(Norm_data[:, 11][150:600], linewidth=4)
    #         plt.show()
    #         plt.savefig(os.path.join('result', 'Abstract' + '_ECG_V1_V2_V3_V4_V5_V6.svg'), dpi=600,
    #                     bbox_inches='tight')
    #         plt.close()
    #         input()

    # Y_ = np.asarray(Y)
    # Y_sum = np.sum(Y_, axis=0)
    # print(Y_sum)

    if datafolder == '../data/CPSC/':
        data = data_slice(data)

    # 10th fold for testing (9th for now)
    X_test = data[labels.strat_fold == test_fold]
    y_test = Y[labels.strat_fold == test_fold]
    # 9th fold for validation (8th for now)
    X_val = data[labels.strat_fold == val_fold]
    y_val = Y[labels.strat_fold == val_fold]
    # rest for training
    X_train = data[labels.strat_fold.isin(train_fold)]
    y_train = Y[labels.strat_fold.isin(train_fold)]

    # Preprocess signal data
    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_datasets(datafolder=None, experiment=None, batch_size=None, train_fold=8, val_fold=9, test_fold=10):
    '''
    Load the final dataset
    '''
    experiment = experiment

    if datafolder == '../data/ptbxl/':
        experiments = {
            'exp0': ('exp0', 'all'),
            'exp1': ('exp1', 'diagnostic'),
            'exp1.1': ('exp1.1', 'subdiagnostic'),
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),
            'exp2': ('exp2', 'form'),
            'exp3': ('exp3', 'rhythm')
        }
        name, task = experiments[experiment]
        X_train, y_train, X_val, y_val, X_test, y_test = \
            kth_fold_data(name, task, datafolder, sampling_frequency=100,  min_samples=0,
                          train_fold=train_fold, val_fold=val_fold, test_fold=test_fold)
        # ded = DownLoadECGData(name, task, datafolder)
        # X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    elif datafolder == '../data/CPSC/':
        X_train, y_train, X_val, y_val, X_test, y_test = \
            kth_fold_data('exp_CPSC', 'all', datafolder, sampling_frequency=100,  min_samples=0,
                          train_fold=train_fold, val_fold=val_fold, test_fold=test_fold)
        # ded = DownLoadECGData('exp_CPSC', 'all', datafolder)
        # X_train, y_train, X_val, y_val, X_test, y_test = ded.preprocess_data()
    elif datafolder == '../data/hf/':
        X_train, y_train, X_val, y_val, X_test, y_test = \
            kth_fold_data('exp_HFHC', 'all', datafolder, sampling_frequency=100,  min_samples=0,
                          train_fold=train_fold, val_fold=val_fold, test_fold=test_fold)

    ds_train = ECGDataset(X_train, y_train)
    ds_val = ECGDataset(X_val, y_val)
    ds_test = ECGDataset(X_test, y_test)

    num_classes = ds_train.num_classes
    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, num_classes
