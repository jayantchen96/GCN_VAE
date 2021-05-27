import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from Wave import createSensorWave


# import random


def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte().bool()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1 - rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


# 划分数据集，shanghai.pkl和shanghai_extra.pkl是原数据集
def split_dataset(filename, split_rate=0.5):
    f = open(filename, 'rb')
    data = pickle.load(f)
    index = int(data.shape[0] * split_rate)
    train_data = data[:index]
    test_data = data[index:]
    fs_train = open(filename.split('.')[0] + '_train' + '.pkl', 'wb')
    fs_test = open(filename.split('.')[0] + '_test' + '.pkl', 'wb')
    pickle.dump(train_data, fs_train)
    pickle.dump(test_data, fs_test)
    return


# 添加异常点的函数，从高斯函数中取样，添加至正常点上使其变成异常点
def add_abnormal(filename, rate=0.05, mu=1, sigma=0.1):
    fo = open(filename, 'rb')
    data = pickle.load(fo)
    # [1900, 10, 3]
    abnormal_num = int(data.shape[0] * rate * data.shape[1])
    np.random.seed(4)
    # 随机某个时刻异常
    index = np.arange(80, data.shape[0] * data.shape[1])
    abnormal_index = np.random.choice(index, abnormal_num, replace=False)
    # abnormal_index = random.sample(range(0, data.shape[0]), abnormal_num)
    np.random.seed(4)
    # 随机某个设备的所有传感器全部异常
    # sensor_index = np.random.randint(data.shape[1], size=abnormal_num)
    # matrix_index = np.zeros((data.shape[0], 2), int)
    label = np.zeros((data.shape[0] * data.shape[1]), int)
    label[abnormal_index] = 1
    label = label.reshape((data.shape[0], data.shape[1]))
    np.random.seed(4)
    abnor_value = np.random.randint(10, 15)
    data = data.reshape((-1, data.shape[2]))
    data[abnormal_index] += abnor_value
    data = data.reshape((label.shape[0], label.shape[1], data.shape[1]))
    # data[abnormal_index] += data[abnormal_index] * 0.1
    print(data.shape, label.shape)
    np.save(filename.split('.')[0] + '_abn' + '.npy', data)
    np.save(filename.split('.')[0] + '_label' + '.npy', label)
    # fs_data = open(filename.split('.')[0] + '_abn' + '.pkl', 'wb')
    # fs_label = open(filename.split('.')[0] + '_label' + '.pkl', 'wb')
    # pickle.dump(data, fs_data, -1)
    # pickle.dump(label, fs_label, -1)
    return data, label


def anomaly_index(data, timestop):
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data.split(',')
    data = list(map(int, data))
    anomaly = np.array([])
    for i in range(len(data) // 2):
        a = data[2 * i]
        b = data[2 * i + 1]
        if a > timestop:
            break
        elif a == timestop:
            anomaly = np.append(anomaly, timestop - 1)
            break
        if b > timestop:
            anomaly = np.concatenate((anomaly, np.arange(a - 1, timestop)))
        else:
            anomaly = np.concatenate((anomaly, np.arange(a - 1, b)))
    return anomaly.astype(int)


def read_anomalies(filename, dataname=None, setname=None, timestop=0):
    ts_df = pd.read_csv(filename, header=0)
    data = ts_df.loc[(ts_df['chan_id'].str.contains(dataname)) & (ts_df['spacecraft'] == setname)]
    labels = np.zeros((timestop, len(data)))
    print(data)
    for i in range(len(data)):
        # index = int(data.iloc[i, 0].split('-')[1])
        # print(index)
        anomaly = data.iloc[i, 2]
        anomaly = anomaly_index(anomaly, timestop)
        labels[anomaly, i] = 1
    print('-----Abnormal points have been read!-----')
    return labels


class SinDataset(Dataset):
    def __init__(self, data_name, time_windows, step, mode="Train"):
        '''
        :param csv_file: 数据集
        :param time_windows: 时间窗长度
        :param step: 时间窗步长
        :param mode: 读取训练集还是测试集，0为训练集，1为测试集
        '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        if data_name == 'shanghai':
            if mode == "Train":
                print('训练模式-余弦')
                normal_data_internal = createSensorWave(timestamp=950, numOfSens=10, numOfFeas=3)
            else:
                print('测试模式-余弦')

        else:

            print('数据集读取失败')
        # 对数据进行归一化，然后再将数据的形状转换回去
        normalized_data_internal = scaler.fit_transform(
            normal_data_internal.reshape(-1, normal_data_internal.shape[-1]))
        normalized_data_internal = normalized_data_internal.reshape(normal_data_internal.shape)
        # 转成Tensor
        torch_data_internal = torch.FloatTensor(normalized_data_internal)

        # 创建邻接矩阵
        adj_path = 'data/wave_conj_dtw.pkl'
        adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
        adj_matrix = torch.FloatTensor(adj_matrix)
        # 变化邻接矩阵
        adj_matrix = torch.eye(adj_matrix.size(0), dtype=torch.int64) + adj_matrix
        # 创建度矩阵
        d_matrix = torch.zeros_like(adj_matrix)
        index = torch.arange(0, adj_matrix.size(0))
        # index = index.reshape(adj_matrix.size(0), 1)
        # index = index.repeat(1, 2)
        d_matrix[index, index] = torch.sum(adj_matrix, dim=1)
        # 改进邻接矩阵
        d_matrix = torch.sqrt(torch.inverse(d_matrix))
        adj_matrix = torch.matmul(torch.matmul(d_matrix, adj_matrix), d_matrix)
        self.inout_seq = []
        self.predict_seq = []
        self.data_length = torch_data_internal.size(0)
        print("original data size:", torch_data_internal.size())
        # 左闭右开
        for i in range(0, self.data_length - time_windows, step):
            data_seq = (torch_data_internal[i:i + time_windows], adj_matrix)
            predict = torch_data_internal[i:i + time_windows + 1]
            # predict = torch_data_internal[i + 1:i + time_windows + 1]
            self.inout_seq.append(data_seq)
            self.predict_seq.append(predict)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        sample = self.inout_seq[index]
        predict = self.predict_seq[index]
        # print("In dataloader....")
        # print(type(sample))
        return sample, predict

    def __len__(self):
        return len(self.predict_seq)


# 基于预测的方法的dataset
class PredictorDataset(Dataset):
    def __init__(self, data_name, time_windows, step, mode="Train"):
        '''
        :param csv_file: 数据集
        :param time_windows: 时间窗长度
        :param step: 时间窗步长
        :param mode: 读取训练集还是测试集，0为训练集，1为测试集
        '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        normal_data_internal = []
        normal_data_external = []
        adj_matrix = []
        if mode == 'Train':
            print('训练模式')
            if data_name == "shanghai":
                data_internal = pickle.load(open('data/shanghai_train.pkl', 'rb'), encoding='utf-8')
                data_external = pickle.load(open('data/shanghai_extra_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                normal_data_external = data_external.copy()
                # 读取邻接矩阵
                # adj_path = 'data/shanghai_conj.pkl'
                adj_path = 'data/shanghai_conj_dtw.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_A":
                data_internal = pickle.load(open('data/SMAP_train/processed/A_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/A_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_D":
                data_internal = pickle.load(open('data/SMAP_train/processed/D_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/D_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_E":
                data_internal = pickle.load(open('data/SMAP_train/processed/E_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/E_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_G":
                data_internal = pickle.load(open('data/SMAP_train/processed/G_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/G_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "msl":
                data_internal = pickle.load(open('data/MSL_train/processed/M_train.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/MSL_train/processed/M_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == 'sin':
                data_internal = pickle.load(open('data/wave.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/wave_conj_dtw.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
        else:
            print('测试模式')
            if data_name == "shanghai":
                data_internal = pickle.load(open('data/shanghai_test_abn.pkl', 'rb'), encoding='utf-8')
                data_external = pickle.load(open('data/shanghai_extra_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                normal_data_external = data_external.copy()
                # 读取邻接矩阵
                # adj_path = 'data/shanghai_conj.pkl'
                adj_path = 'data/shanghai_conj_dtw.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_A":
                data_internal = pickle.load(open('data/SMAP_test/processed/A_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/A_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_D":
                data_internal = pickle.load(open('data/SMAP_test/processed/D_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/D_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_E":
                data_internal = pickle.load(open('data/SMAP_test/processed/E_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/E_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "smap_G":
                data_internal = pickle.load(open('data/SMAP_test/processed/G_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/SMAP_train/processed/G_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "msl":
                data_internal = pickle.load(open('data/MSL_test/processed/M_test.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/MSL_train/processed/M_conj_dtw90.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
            if data_name == "sin":
                data_internal = pickle.load(open('data/wave_abn.pkl', 'rb'), encoding='utf-8')
                normal_data_internal = data_internal.copy()
                # 读取邻接矩阵
                adj_path = 'data/wave_conj_dtw.pkl'
                adj_matrix = pickle.load(open(adj_path, 'rb'), encoding='utf-8')
                adj_matrix = torch.FloatTensor(adj_matrix)
        # 对数据进行归一化，然后再将数据的形状转换回去
        # for i in range(normal_data_internal.shape[-1]):
        #     normal_data_internal[:,:,i] = scaler.fit_transform(normal_data_internal[:,:,i])
        normalized_data_internal = scaler.fit_transform(
            normal_data_internal.reshape(-1, normal_data_internal.shape[-1]))
        normalized_data_internal = normalized_data_internal.reshape(normal_data_internal.shape)
        # 转成Tensor
        torch_data_internal = torch.FloatTensor(normalized_data_internal)
        if len(normal_data_external) != 0:
            normalized_data_external = scaler.fit_transform(
                normal_data_external.reshape(-1, normal_data_external.shape[-1]))
            normalized_data_external = normalized_data_external.reshape(normal_data_external.shape)
            torch_data_external = torch.FloatTensor(normalized_data_external)
        # 变化邻接矩阵
        # adj_matrix = torch.eye(adj_matrix.size(0), dtype=torch.int64) + adj_matrix
        adj_matrix = torch.ones_like(adj_matrix)
        # 创建度矩阵
        d_matrix = torch.zeros_like(adj_matrix)
        index = torch.arange(0, adj_matrix.size(0))
        # index = index.reshape(adj_matrix.size(0), 1)
        # index = index.repeat(1, 2)
        d_matrix[index, index] = torch.sum(adj_matrix, dim=1)
        # 改进邻接矩阵
        d_matrix = torch.sqrt(torch.inverse(d_matrix))
        adj_matrix = torch.matmul(torch.matmul(d_matrix, adj_matrix), d_matrix)
        self.inout_seq = []
        self.predict_seq = []
        self.data_length = torch_data_internal.size(0)
        print("dataset:", data_name)
        print("original data size:", torch_data_internal.size())
        # 左闭右开
        for i in range(0, self.data_length - time_windows, step):
            data_seq = (torch_data_internal[i:i + time_windows], adj_matrix)
            predict = torch_data_internal[i:i + time_windows + 1]
            # data_seq = (torch_data_internal[i:i + time_windows], adj_matrix, torch_data_external[i:i + time_windows])
            # predict = (torch_data_internal[i:i + time_windows + 1], adj_matrix,
            #            torch_data_external[i:i + time_windows + 1])
            # predict = torch_data_internal[i+1:i + time_windows + 1]
            # predict = torch_data_internal[i+time_windows]
            self.inout_seq.append(data_seq)
            self.predict_seq.append(predict)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        sample = self.inout_seq[index]
        predict = self.predict_seq[index]
        return sample, predict

    def __len__(self):
        return len(self.predict_seq)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def plotFigure(filename):
    fo = open(filename, 'rb')
    data = pickle.load(fo)
    data = data[:950, 9, :]
    # print(data)
    x = np.arange(950)
    plt.plot(x, data)
    plt.show()


# split_dataset('data/shanghai.pkl')
# split_dataset('data/shanghai_extra.pkl')
# add_abnormal('data/shanghai_test.pkl')
# plotFigure('data/wave_abn.pkl')
# fo = open("data/wave_label.pkl", 'rb')
# labels = pickle.load(fo)
# print(np.where(labels[:8,:]==1)[0])
# plotFigure('data/MSL_train/processed/M_train.pkl')
# plotFigure('data/SMAP_test/processed/D_test.pkl')

if __name__ == '__main__':
    print("hello")
