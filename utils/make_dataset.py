import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
from utils import seed_everything


class MyDataset(data.Dataset):
    def __init__(self, dataset, time_windows, step, is_train=True):
        seed_everything()

        self.is_train = is_train

        # 读取数据集源文件
        data_path = '../data/processed'
        monitor_data = np.load(os.path.join(data_path, f'{dataset}.npy'), 'r')
        adj_data = np.load(os.path.join(data_path, f'{dataset}_adj.npy'), 'r')

        mask5_data = np.load(os.path.join(data_path, f'{dataset}_mask_5.npy'), 'r')
        mask10_data = np.load(os.path.join(data_path, f'{dataset}_mask_10.npy'), 'r')
        mask30_data = np.load(os.path.join(data_path, f'{dataset}_mask_30.npy'), 'r')

        # 对 monitor data 归一化
        scaler = MinMaxScaler(feature_range=(0, 1))

        ori_shape = monitor_data.shape

        monitor_data = monitor_data.reshape(-1, monitor_data.shape[-1])
        monitor_data = scaler.fit_transform(monitor_data).reshape(ori_shape)

        # 划分训练集、测试集
        train_len = int(monitor_data.shape[0] * 2 / 3)

        if self.is_train:
            monitor_data = monitor_data[:train_len]
            mask5_data = mask5_data[:train_len]
            mask10_data = mask10_data[:train_len]
            mask30_data = mask30_data[:train_len]
        else:
            monitor_data = monitor_data[train_len:]
            mask5_data = mask5_data[train_len:]
            mask10_data = mask10_data[train_len:]
            mask30_data = mask30_data[train_len:]

        self.timestamps, self.num_devices, self.num_features = monitor_data.shape

        self.data = []
        self.mask5 = []
        self.mask10 = []
        self.mask30 = []
        self.adj_data = adj_data

        # 滑动时间窗
        for i in range(0, self.timestamps - time_windows, step):
            self.data.append(monitor_data[i: i + time_windows])
            self.mask5.append(mask5_data[i: i + time_windows])
            self.mask10.append(mask10_data[i: i + time_windows])
            self.mask30.append((mask30_data[i: i + time_windows]))

        print("=" * 20)
        print("训练集" if is_train else "测试集")
        print("总时间长度:", self.timestamps)
        print("时间窗:", time_windows)
        print("总设备数:", self.num_devices)
        print("设备传感器个数:", self.num_features)
        print("样本数:", len(self.data))
        print("=" * 20)

    def __getitem__(self, index):

        sample = self.data[index]
        mask5 = self.mask5[index]
        mask10 = self.mask10[index]
        mask30 = self.mask30[index]

        return sample, self.adj_data, mask5, mask10, mask30

    def __len__(self):

        return len(self.data)


if __name__ == '__main__':
    dataset = MyDataset(dataset='prsa', time_windows=10, step=1, is_train=True)
