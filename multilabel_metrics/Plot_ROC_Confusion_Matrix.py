import numpy as np, os, sys, json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, auc
from itertools import cycle


def plot_Multiclass_ROC_curve(test_targets, test_outputs, num_classes, label_names, savename):
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(11, 11))

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_targets[:, i], test_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(num_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=8,
    )

    # classes = ['IAVB', 'AF', 'CLBBB', 'CRBBB', 'Norm', 'PAC', 'STD', 'STE', 'PVC']
    colors = cycle(
        ["aqua", "darkorange", "darkorchid", "palegreen", "palevioletred", "pink", "cornflowerblue", "purple",
         "salmon"])
    for class_id, color in zip(range(num_classes), colors):
        RocCurveDisplay.from_predictions(
            test_targets[:, class_id],
            test_outputs[:, class_id],
            name=f"ROC curve for {label_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 8),
            chance_level_kw={"linewidth": 4},
            linewidth=4,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves across Multiple Categories")
    plt.legend(fontsize="14")
    plt.show()
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()


# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A


def plot_modified_confusion_matrix(confusion_matrix, savename):
    # ECG abnormalities
    classes = ['IAVB', 'AF', 'CLBBB', 'CRBBB', 'Norm', 'PAC', 'STD', 'STE', 'PVC']

    # Normalize by row 鉴于多种心脏疾病类别不均衡,对混淆矩阵做归一化更直观展示分类器性能
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # # plot confusion_matrix style A
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.imshow(cm_normalized, cmap=plt.cm.Blues)
    # fig.colorbar(cax)
    # ax.xaxis.set_major_locator(MultipleLocator(1))  # 将x主刻度标签设置为1的倍数,对应于每一个class
    # ax.yaxis.set_major_locator(MultipleLocator(1))  # 将y主刻度标签设置为1的倍数,对应于每一个class
    # # set the fontsize of label.
    # for label in plt.gca().xaxis.get_ticklabels():
    #     label.set_fontsize(10)
    # for i in range(cm_normalized.shape[0]):
    #     ax.text(i, i, str('%.1f' % (cm_normalized[i, i])), fontsize=6, va='center', ha='center')
    # ax.set_xticklabels([''] + classes, rotation=90)
    # ax.set_yticklabels([''] + classes)
    # plt.ylabel('True Labels')
    # plt.xlabel('Predicted Labels')
    # plt.title('Modified Confusion Matrix')
    # plt.savefig(savename, dpi=300, bbox_inches='tight')

    # plot confusion_matrix style B
    # plt.rcParams.update({'font.size': 10})
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = np.max(cm_normalized) / 2.0
    # print(np.max(cm_normalized))
    # print(thresh)
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (cm_normalized.size, 2))
    for i, j in iters:
        # if cm_normalized[i, j] > thresh:
        #     plt.text(j, i, str('%.1f' % (cm_normalized[i, j])), color='red', fontsize=6, va='center', ha='center')  # 显示对应的数字
        # else:
        #     plt.text(j, i, str('%.1f' % (cm_normalized[i, j])), fontsize=6, va='center', ha='center')  # 显示对应的数字
        plt.text(j, i, str('%.1f' % (cm_normalized[i, j])), fontsize=8, va='center', ha='center')  # 显示对应的数字

    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()
