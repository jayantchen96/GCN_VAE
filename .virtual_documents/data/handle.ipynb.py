import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
import os
import math
import random
import datetime
from tqdm import tqdm
get_ipython().run_line_magic("matplotlib", " widget")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(seed=10086)


# 输出目录
output_dir = 'processed'


miss_rates = [5, 10, 30]


# (MCAR) Missing completely at random
def generate_mcar_mask(samples, ratio=0.1):
    assert isinstance(samples, np.ndarray)

    mask_matrix = np.random.choice([0, 1], size=samples.shape, p=[ratio, 1 - ratio])

    return mask_matrix


train_proportion = 2. / 3
test_proportion = 1. - train_proportion


ghg_dir = 'GHG'
num_files = len(os.listdir(ghg_dir))
print(f'GHG文件夹中有{num_files}个子文件')


# 随机选指定数量的文件当devices
num_devices = 7
indexes = np.random.choice(num_files, num_devices, replace=False)
indexes.sort()
indexes


# 读取 dat 文件 转成numpy数组
ghg_list = []
for index in indexes:    
    filepath = os.path.join(ghg_dir, 'ghg.gid.site' + str(index).zfill(4) + '.dat')
    df = pd.read_csv(filepath, sep = " ", header=None, dtype=np.float32) 
    data = df.values
    data = np.transpose(data, (1, 0))
    ghg_list.append(data)

ghg_arr = np.stack(ghg_list, axis=1)


# (seq_len, num_devices, num_features)
ghg_arr.shape


# 保存成 npy 文件
np.save(os.path.join(output_dir, 'ghg.npy'), ghg_arr)


# 加载数组
ghg_arr = np.load(os.path.join(output_dir, 'ghg.npy'))
print(ghg_arr.shape)


# 生成 mask
for rate in miss_rates:
    mask = generate_mcar_mask(ghg_arr, ratio=rate / 100.)
    np.save(os.path.join(output_dir, f'ghg_mask_{rate}.npy'), mask)


mask = generate_mcar_mask(np.random.randn(35064, 12, 6), ratio=0.05)
np.save(os.path.join(output_dir, f'prsa_mask_5.npy'), mask)


def temporal_KNN(seq, k_num):
    """
    时间KNN
    :param seq: 含有NAN的时间序列
    :param k_num: K近邻数
    :return: 补全的时间序列
    """
    ret = np.copy(seq)
    for index in range(len(seq)):
        if ~np.isnan(seq[index]):
            continue

        candi = []

        offset = 1
        while offset < max(index, len(seq) - index) and len(candi) < k_num:
            if index - offset >= 0 and ~np.isnan(ret[index - offset]):
                candi.append((offset, ret[index - offset]))
            if index + offset < len(seq) and ~np.isnan(ret[index + offset]):
                candi.append((offset, ret[index + offset]))
            offset += 1

        if len(candi) > 0:
            candi.sort(key=lambda elem: elem[0])
            p = 0.
            weights = 0.
            mean_val = 0.
            for x in range(min(len(candi), k_num)):
                # 加权算法
                weights += 1. / candi[x][0]
                p += candi[x][1] / candi[x][0]

                # 平均值算法
                # mean_val += candi[x][1]

            # seq[index] = mean_val / min(len(candi), k_num)
            seq[index] = p / weights

    return seq


prsa_dir = 'PRSA'
num_files = len(os.listdir(prsa_dir))
print(f'PRSA文件夹中有{num_files}个子文件')


# 用KNN补全NAN做预处理
monitor_list = []
with tqdm(total=72) as pbar:
    for filename in os.listdir(prsa_dir):
        if filename.startswith('PRSA'):
            df = pd.read_csv(os.path.join(prsa_dir, filename), sep=',')
            monitor_arr = df.iloc[:, 5:11].values
    #         extra_arr = df.iloc[:, 11:17].values
            for i in range(6):
                monitor_arr[:,i] = temporal_KNN(monitor_arr[:,i], k_num=2)
                pbar.update(1)

            monitor_list.append(monitor_arr)

monitor_data = np.stack(monitor_list, axis=1)
print(monitor_data.shape)
print('处理后的数据含有的缺失值数量:', np.isnan(monitor_data).sum())
        


# 保存成 npy 文件
np.save(os.path.join(output_dir, 'prsa.npy'), monitor_data)


# 生成 mask
for rate in miss_rates:
    mask = generate_mcar_mask(monitor_data, ratio=rate / 100.)
    np.save(os.path.join(output_dir, f'prsa_mask_{rate}.npy'), mask)


mm = np.load(os.path.join(output_dir, f'prsa_mask_30.npy'))


mm.shape



