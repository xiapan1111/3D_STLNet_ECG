import os
import pandas as pd
import wfdb
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from scipy.io import loadmat
from stratisfy import stratisfy_df

output_folder = '/home/data/ecg_benchmarking_ptbxl/official_experiments/data/hf/'
output_datafolder_100 = output_folder + '/records100/'
output_datafolder_500 = output_folder + '/records500/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(output_datafolder_100):
    os.makedirs(output_datafolder_100)
if not os.path.exists(output_datafolder_500):
    os.makedirs(output_datafolder_500)


def store_as_wfdb(signame, data, sigfolder, fs):
    channel_itos = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    wfdb.wrsamp(signame,
                fs=fs,
                sig_name=channel_itos,
                p_signal=data,
                units=['mV'] * len(channel_itos),
                fmt=['16'] * len(channel_itos),
                write_dir=sigfolder)


# df_reference = pd.read_csv('/home/data/ecg_benchmarking_ptbxl/official_experiments/data/tmp_data/REFERENCE.csv')
# read txt method oneX:\Drive\ecg_benchmarking_ptbxl\official_experiments\data\hf
file_label = open("/home/data/ecg_benchmarking_ptbxl/official_experiments/data/hf/hf_round2_label.txt")
df_reference = file_label.readlines()

# label_dict = {'': 'NORM', 2: 'AFIB', 3: '1AVB', 4: 'CLBBB', 5: 'CRBBB', 6: 'PAC', 7: 'VPC', 8: 'STD_', 9: 'STE_'}

label_dict = {'QRS低电压': 'LQRSV', '电轴右偏': 'RAD', '起搏心律': 'PaceR', 'T波改变': 'TAb', '电轴左偏': 'LAD', '心房颤动': 'AF',
              '非特异性ST段异常': 'SpST_', '下壁异常Q波': 'QAb', '前间壁R波递增不良': 'FRA', 'ST段改变': 'ST_', '一度房室传导阻滞': '1AVB',
              '左束支传导阻滞': 'LBBB', '右束支传导阻滞': 'RBBB', '完全性左束支传导阻滞': 'CLBBB', '左前分支传导阻滞': 'LFnBB',
              '右心房扩大': 'RKD', '短PR间期': 'SPRI', '左心室高电压': 'LVHV', '窦性心动过缓': 'SB', '早期复极化': 'EDP',
              '窦性心律': 'SRH', '融合波': 'FuWa', 'ST-T改变': 'ST_T', '非特异性ST段与T波异常': 'SpST_T', '快心室率': 'FVR', '非特异性T波异常': 'NSTAb',
              '室性早搏': 'PVC', '房性早搏': 'PAC', '窦性心律不齐': 'SA', '完全性右束支传导阻滞': 'CRBBB', '窦性心动过速': 'SR', '不完全性右束支传导阻滞': 'ICRBBB',
              '顺钟向转位': 'SFZ', '逆钟向转位': 'NSZ'}

data = {'ecg_id': [], 'filename': [], 'validation': [], 'age': [], 'sex': [], 'scp_codes': []}

ecg_counter = 0
filenames = os.listdir('/home/data/ecg_benchmarking_ptbxl/official_experiments/data/hf/hf_round2_train')
for filename in tqdm(filenames):
    if filename.split('.')[1] == 'txt':
        name = filename.split('.')[0]
        file_data = open('/home/data/ecg_benchmarking_ptbxl/official_experiments/data/hf/hf_round2_train/' + filename)
        data_str = file_data.readline()
        I_lead = []
        II_lead = []
        V1_lead = []
        V2_lead = []
        V3_lead = []
        V4_lead = []
        V5_lead = []
        V6_lead = []
        while data_str:
            data_str = file_data.readline()
            if data_str != '':
                data_str = data_str.split(' ')
                I_lead.append(float(data_str[0].strip()))
                II_lead.append(float(data_str[1].strip()))
                V1_lead.append(float(data_str[2].strip()))
                V2_lead.append(float(data_str[3].strip()))
                V3_lead.append(float(data_str[4].strip()))
                V4_lead.append(float(data_str[5].strip()))
                V5_lead.append(float(data_str[6].strip()))
                V6_lead.append(float(data_str[7].strip()))
                # input()

        sig = np.vstack((I_lead, II_lead, V1_lead, V2_lead, V3_lead, V4_lead, V5_lead, V6_lead))
        # print(sig.shape)
        for j in range(len(df_reference)):
            label_str = df_reference[j].split('\t')
            del (label_str[-1])
            if label_str[0].strip().split('.')[0] == name:
                # age, sex = label_str[1].strip(), label_str[2].strip()
                age, sex = 50, 'MALE'
                data['ecg_id'].append(ecg_counter + 1)
                data['filename'].append(name)
                data['validation'].append(False)
                data['age'].append(int(age))

                data['sex'].append(1 if sex == 'MALE' else 0)
                labels = [label_str[i + 3].strip() for i in range(len(label_str) - 3)]
                # print(labels)
                # print({label_dict[key]: 100 for key in labels})

                data['scp_codes'].append({label_dict[key]: 100 for key in labels})
                store_as_wfdb(str(ecg_counter + 1), sig.T, output_datafolder_500, 500)
                down_sig = np.array([zoom(channel, .2) for channel in sig])
                store_as_wfdb(str(ecg_counter + 1), down_sig.T, output_datafolder_100, 100)
                # input()
                ecg_counter += 1

df = pd.DataFrame(data)
df['patient_id'] = df.ecg_id
df = stratisfy_df(df, 'strat_fold')
df.to_csv(output_folder + 'hfhc_database.csv')


