from operator import itemgetter
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
from multilabel_metrics.multilabel_metric import *
import heapq


def train_threshold(data_y, pred):
    # print("data_y.shape", data_y.shape)
    # print("pred.shape", pred.shape)

    data_num = data_y.shape[0]
    label_num = data_y.shape[1]
    threshold = np.zeros([data_num])

    for i in range(data_num):
        pred_i = pred[i, :]
        # print("pred_i", pred_i)
        # x_i = data_x[i, :]
        y_i = data_y[i, :]
        # print("y_i", y_i)
        tup_list = []
        for j in range(len(pred_i)):
            tup_list.append((pred_i[j], y_i[j]))

        tup_list = sorted(tup_list, key=itemgetter(0))
        min_val = label_num
        for j in range(len(tup_list) - 1):
            val_measure = 0

            for k in range(j + 1):
                if(tup_list[k][1] == 1):
                    val_measure = val_measure + 1
            for k in range(j + 1, len(tup_list)):
                if(tup_list[k][1] == 0):
                    val_measure = val_measure + 1

            if val_measure < min_val:
                min_val = val_measure
                threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2
        # print("threshold[i]", threshold[i])
        # input()

    linreg = LinearRegression()
    linreg.fit(pred, threshold)
    joblib.dump(linreg, "./multilabel_metrics/linear_model.pkl")


def threshold_predict(pred):
    # print("pred.shape", pred.shape)
    linreg = joblib.load('./multilabel_metrics/linear_model.pkl')
    threshold = linreg.predict(pred)
    # print("threshold.shape", threshold.shape)
    y_pred = ((pred.T - threshold.T) > 0).T
    # print("y_pred.shape", y_pred.shape)

    #translate bool to int
    y_pred = y_pred + 0
    return y_pred, threshold


def train_threshold_class(epoch, data_y, pred):
    data_y = data_y.T
    pred = pred.T

    data_num = data_y.shape[0]
    label_num = data_y.shape[1]
    threshold = 0.3 * np.ones([data_num])   # initial class threshold
    if epoch == 16:
        for i in range(data_num):
            print('    {}/{}...'.format(i + 1, data_num))
            pred_i = pred[i, :]
            # print("pred_i", pred_i)
            # x_i = data_x[i, :]
            y_i = data_y[i, :]
            # print("y_i", y_i)
            tup_list = []
            for j in range(len(pred_i)):
                tup_list.append((pred_i[j], y_i[j]))

            tup_list = sorted(tup_list, key=itemgetter(0))
            min_val = label_num
            for j in range(len(tup_list) - 1):
                val_measure = 0

                for k in range(j + 1):
                    if (tup_list[k][1] == 1):
                        val_measure = val_measure + 1
                for k in range(j + 1, len(tup_list)):
                    if (tup_list[k][1] == 0):
                        val_measure = val_measure + 1

                if val_measure < min_val:
                    min_val = val_measure
                    threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2
            # print("threshold[i]", threshold[i])
            # input()

    return threshold


def One_threshold(val_targets, val_outputs, num_classes):
    hamming_loss_bufer = []
    for threshold_index in range(0, 11):
        threshold = 0.1 * threshold_index
        val_predicts = np.zeros((len(val_outputs), num_classes), dtype=np.bool)
        for instance in range(len(val_outputs)):
            val_predicts[instance] = np.array(
                [True if val_outputs[instance][j] >= threshold else False
                 for j in range(len(val_outputs[instance]))]).astype(np.bool)
        hamming_loss = hammingLoss(val_targets, val_predicts)
        hamming_loss_bufer.append(hamming_loss)
    hamming_loss_min = min(hamming_loss_bufer)
    threshold_one_optimal = hamming_loss_bufer.index(hamming_loss_min) * 0.1
    threshold_class_optimal = threshold_one_optimal * np.ones(num_classes)

    return threshold_class_optimal


# grid search algorithm hamming loss
def specclass_threshold(labels, pred_outputs, num_class):
    # find optimal threshold
    threshold_class_optimal = np.zeros(num_class)
    for label in range(num_class):
        hamming_loss_bufer = []
        threshold_class = 0.0 * np.ones(num_class)  # initial class threshold
        for threshold_index in range(0, 11):
            threshold_class[label] = 0.1 * threshold_index
            # print("threshold_class[label]", threshold_class)
            label_predicts = np.zeros((len(pred_outputs), num_class), dtype=np.bool)
            for instance in range(len(pred_outputs)):
                label_predicts[instance] = np.array(
                    [True if pred_outputs[instance][j] >= threshold_class[j] else False
                     for j in range(len(pred_outputs[instance]))]).astype(np.bool)

            # calculate hamming loss
            hamming_loss = hammingLoss(labels, label_predicts)
            hamming_loss_bufer.append(hamming_loss)
        hamming_loss_min = min(hamming_loss_bufer)
        threshold_class_optimal[label] = hamming_loss_bufer.index(hamming_loss_min) * 0.1
    # print("threshold_class_optimal", threshold_class_optimal)

    return threshold_class_optimal


def calculate_cardinality(val_labels):
    samples_labels_sum_total = 0.0
    for i in range(len(val_labels)):
        samples_labels_sum_total += np.sum(val_labels[i])
    cardinality = samples_labels_sum_total / len(val_labels)
    print(cardinality)

    return round(cardinality)


def based_cardinality_predict(test_outputs, num_class, label_cardinality):
    test_predicts = np.zeros((len(test_outputs), num_class), dtype=np.bool)
    for i in range(len(test_outputs)):
        indexs = heapq.nlargest(label_cardinality, range(num_class), test_outputs[i].__getitem__)
        # print(indexs)
        for index in range(len(indexs)):
            # print(indexs[index])
            test_predicts[i][indexs[index]] = 1

    return test_predicts
