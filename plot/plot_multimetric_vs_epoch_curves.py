import numpy as np, os
import matplotlib.pyplot as plt

result_path = './result/'
plot_path = './plot/'

def read_reslts(result_path):
    traing_loss_dict = {}
    traing_auc_dict = {}
    val_auc_dict = {}
    val_acc_dict = {}
    val_f1_dict = {}
    val_hamm_dict = {}

    dirs = os.listdir(result_path)
    for file in dirs:
        root_file, _ = os.path.splitext(file)
        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'training_loss':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    traing_loss_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    traing_loss_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    traing_loss_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    traing_loss_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    traing_loss_dict['hf'] = f4.read()

        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'training_auc':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    traing_auc_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    traing_auc_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    traing_auc_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    traing_auc_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    traing_auc_dict['hf'] = f4.read()

        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'val_auc':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    val_auc_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    val_auc_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    val_auc_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    val_auc_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    val_auc_dict['hf'] = f4.read()

        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'val_acc':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    val_acc_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    val_acc_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    val_acc_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    val_acc_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    val_acc_dict['hf'] = f4.read()

        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'val_f1':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    val_f1_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    val_f1_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    val_f1_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    val_f1_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    val_f1_dict['hf'] = f4.read()

        if root_file.split('_')[0] + '_' + root_file.split('_')[1] == 'val_hamm':
            if root_file.split('_')[2] == 'exp3':
                with open(result_path + file, 'r') as f0:
                    val_hamm_dict['exp3'] = f0.read()
            if root_file.split('_')[2] == 'exp1.1':
                with open(result_path + file, 'r') as f1:
                    val_hamm_dict['exp1.1'] = f1.read()
            if root_file.split('_')[2] == 'exp1.1.1':
                with open(result_path + file, 'r') as f2:
                    val_hamm_dict['exp1.1.1'] = f2.read()
            if root_file.split('_')[2] == 'cpsc':
                with open(result_path + file, 'r') as f3:
                    val_hamm_dict['cpsc'] = f3.read()
            if root_file.split('_')[2] == 'hf':
                with open(result_path + file, 'r') as f4:
                    val_hamm_dict['hf'] = f4.read()

    return traing_loss_dict, traing_auc_dict, val_auc_dict, val_acc_dict, val_f1_dict, val_hamm_dict


def plot_multimetric_vs_epoch_curves(result_path, plot_path):
    traing_loss_dict, traing_auc_dict, val_auc_dict, val_acc_dict, val_f1_dict, val_hamm_dict = read_reslts(result_path)

    plt.rcParams.update({'font.size': 16})

    x_axis = np.arange(1, 51)

    # darkorchid, cornflowerblue, purple, salmon, palevioletred

    # plot training loss
    traing_loss_exp3 = traing_loss_dict["exp3"][1:-1].split(', ')
    traing_loss_exp3 = np.asfarray(traing_loss_exp3, float)
    traing_loss_exp1_1 = traing_loss_dict["exp1.1"][1:-1].split(', ')
    traing_loss_exp1_1 = np.asfarray(traing_loss_exp1_1, float)
    traing_loss_exp1_1_1 = traing_loss_dict["exp1.1.1"][1:-1].split(', ')
    traing_loss_exp1_1_1 = np.asfarray(traing_loss_exp1_1_1, float)
    traing_loss_cpsc = traing_loss_dict["cpsc"][1:-1].split(', ')
    traing_loss_cpsc = np.asfarray(traing_loss_cpsc, float)
    traing_loss_HFHC = traing_loss_dict["hf"][1:-1].split(', ')
    traing_loss_HFHC = np.asfarray(traing_loss_HFHC, float)

    plt.figure(1)
    plt.plot(x_axis, traing_loss_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_loss_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_loss_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_loss_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_loss_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title("Training loss on multiple tasks")
    plt.legend(fontsize="10")
    plt.ylim(-0.02, 0.5)
    savename = os.path.join(plot_path, 'training_loss.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()

    # plot training auc
    traing_auc_exp3 = traing_auc_dict["exp3"][1:-1].split(', ')
    traing_auc_exp3 = np.asfarray(traing_auc_exp3, float)
    traing_auc_exp1_1 = traing_auc_dict["exp1.1"][1:-1].split(', ')
    traing_auc_exp1_1 = np.asfarray(traing_auc_exp1_1, float)
    traing_auc_exp1_1_1 = traing_auc_dict["exp1.1.1"][1:-1].split(', ')
    traing_auc_exp1_1_1 = np.asfarray(traing_auc_exp1_1_1, float)
    traing_auc_cpsc = traing_auc_dict["cpsc"][1:-1].split(', ')
    traing_auc_cpsc = np.asfarray(traing_auc_cpsc, float)
    traing_auc_HFHC = traing_auc_dict["hf"][1:-1].split(', ')
    traing_auc_HFHC = np.asfarray(traing_auc_HFHC, float)

    plt.figure(1)
    plt.plot(x_axis, traing_auc_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_auc_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_auc_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_auc_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, traing_auc_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Training AUROC')
    plt.title("Training AUROC on multiple tasks")
    plt.legend(fontsize="10")
    savename = os.path.join(plot_path, 'training_AUROC.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()

    # plot valid auc
    val_auc_exp3 = val_auc_dict["exp3"][1:-1].split(', ')
    val_auc_exp3 = np.asfarray(val_auc_exp3, float)
    val_auc_exp1_1 = val_auc_dict["exp1.1"][1:-1].split(', ')
    val_auc_exp1_1 = np.asfarray(val_auc_exp1_1, float)
    val_auc_exp1_1_1 = val_auc_dict["exp1.1.1"][1:-1].split(', ')
    val_auc_exp1_1_1 = np.asfarray(val_auc_exp1_1_1, float)
    val_auc_cpsc = val_auc_dict["cpsc"][1:-1].split(', ')
    val_auc_cpsc = np.asfarray(val_auc_cpsc, float)
    val_auc_HFHC = val_auc_dict["hf"][1:-1].split(', ')
    val_auc_HFHC = np.asfarray(val_auc_HFHC, float)

    plt.figure(1)
    plt.plot(x_axis, val_auc_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_auc_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_auc_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_auc_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_auc_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation AUROC')
    plt.title("Validation AUROC on multiple tasks")
    plt.legend(fontsize="10")
    plt.ylim(0.65, 1.0)
    savename = os.path.join(plot_path, 'Validation_AUROC.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()

    # plot valid acc
    val_acc_exp3 = val_acc_dict["exp3"][1:-1].split(', ')
    val_acc_exp3 = np.asfarray(val_acc_exp3, float) * 100
    val_acc_exp1_1 = val_acc_dict["exp1.1"][1:-1].split(', ')
    val_acc_exp1_1 = np.asfarray(val_acc_exp1_1, float) * 100
    val_acc_exp1_1_1 = val_acc_dict["exp1.1.1"][1:-1].split(', ')
    val_acc_exp1_1_1 = np.asfarray(val_acc_exp1_1_1, float) * 100
    val_acc_cpsc = val_acc_dict["cpsc"][1:-1].split(', ')
    val_acc_cpsc = np.asfarray(val_acc_cpsc, float) * 100
    val_acc_HFHC = val_acc_dict["hf"][1:-1].split(', ')
    val_acc_HFHC = np.asfarray(val_acc_HFHC, float) * 100

    plt.figure(1)
    plt.plot(x_axis, val_acc_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_acc_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_acc_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_acc_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_acc_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation accuracy (%)')
    plt.title("Validation accuracy on multiple tasks")
    plt.legend(fontsize="10")
    plt.ylim(10, 100)
    savename = os.path.join(plot_path, 'Validation_Accuracy.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()

    # plot valid f1
    val_f1_exp3 = val_f1_dict["exp3"][1:-1].split(', ')
    val_f1_exp3 = np.asfarray(val_f1_exp3, float) * 100
    val_f1_exp1_1 = val_f1_dict["exp1.1"][1:-1].split(', ')
    val_f1_exp1_1 = np.asfarray(val_f1_exp1_1, float) * 100
    val_f1_exp1_1_1 = val_f1_dict["exp1.1.1"][1:-1].split(', ')
    val_f1_exp1_1_1 = np.asfarray(val_f1_exp1_1_1, float) * 100
    val_f1_cpsc = val_f1_dict["cpsc"][1:-1].split(', ')
    val_f1_cpsc = np.asfarray(val_f1_cpsc, float) * 100
    val_f1_HFHC = val_f1_dict["hf"][1:-1].split(', ')
    val_f1_HFHC = np.asfarray(val_f1_HFHC, float) * 100

    plt.figure(1)
    plt.plot(x_axis, val_f1_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_f1_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_f1_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_f1_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_f1_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation F1 score (%)')
    plt.title("Validation F1 score on multiple tasks")
    plt.legend(fontsize="10")
    plt.ylim(25, 100)
    savename = os.path.join(plot_path, 'Validation_F1.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()

    # plot valid hamm
    val_hamm_exp3 = val_hamm_dict["exp3"][1:-1].split(', ')
    val_hamm_exp3 = np.asfarray(val_hamm_exp3, float)
    val_hamm_exp1_1 = val_hamm_dict["exp1.1"][1:-1].split(', ')
    val_hamm_exp1_1 = np.asfarray(val_hamm_exp1_1, float)
    val_hamm_exp1_1_1 = val_hamm_dict["exp1.1.1"][1:-1].split(', ')
    val_hamm_exp1_1_1 = np.asfarray(val_hamm_exp1_1_1, float)
    val_hamm_cpsc = val_hamm_dict["cpsc"][1:-1].split(', ')
    val_hamm_cpsc = np.asfarray(val_hamm_cpsc, float)
    val_hamm_HFHC = val_hamm_dict["hf"][1:-1].split(', ')
    val_hamm_HFHC = np.asfarray(val_hamm_HFHC, float)

    plt.figure(1)
    plt.plot(x_axis, val_hamm_exp3, label="rhythm", color="darkorchid", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_hamm_exp1_1, label="subclass", color="cornflowerblue", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_hamm_exp1_1_1, label="superclass", color="purple", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_hamm_cpsc, label="CPSC", color="salmon", linestyle="solid", linewidth=3)
    plt.plot(x_axis, val_hamm_HFHC, label="HFHC", color="palevioletred", linestyle="solid", linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Validation hamm loss')
    plt.title("Validation hamm loss on multiple tasks")
    plt.legend(fontsize="10")
    # plt.ylim(0.00, 0.20)
    savename = os.path.join(plot_path, 'Validation_Hamm.svg')
    plt.savefig(savename, dpi=1200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    plot_multimetric_vs_epoch_curves(result_path, plot_path)
