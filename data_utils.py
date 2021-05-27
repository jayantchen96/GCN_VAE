#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# 移动平均图
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.ewma(timeSeries, span=size)

    timeSeries.plot(color='blue', label='Original')
    rol_weighted_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()


def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()


def read_SAMP(filename, is_label):
    with open(filename, 'rb') as fopen:
        smap_data = pickle.load(fopen, encoding='bytes')
    print("data shape: {0}".format(smap_data.shape))
    print("data type: {0}".format(smap_data.dtype))
    # (timestamp, features)
    if is_label:
        smap_data = smap_data.astype(np.int)
    return smap_data


# 读取KPI数据集
def read_KPI(filename1=None, filename2=None):
    # load train dataset
    # train_df = pd.read_csv(filename1)
    # print(train_df.head())
    # train_vals = train_df.value.values[:]
    # train_labels = train_df.label.values
    # train_timestamp = train_df.timestamp.values
    # load test dataset
    test_df = pd.HDFStore(filename2).get('data')
    test_vals = test_df.value.values[:12000:2]
    test_labels = test_df.label.values[:12000:2]
    test_timestamp = test_df.timestamp.values[:12000:2]
    # train dataset and test data total 6000 timestamps
    train_log, test_log = np.log10(test_vals[:1000] + 1), np.log10(test_vals[1000:4300] + 1)
    return train_log, test_log, test_vals[1000:4300], test_labels[1000:4300], test_timestamp[1000:4300]


def read_Crackmeter(filename, is_train):
    ts_df = pd.read_csv(filename, header=None)
    data_vals = ts_df.iloc[:, 2].values.astype(np.float)
    data_timestamps = ts_df.iloc[:, 1].values
    data_labels = None
    if is_train:
        data_vals = data_vals[-1000:]
        data_timestamps = data_timestamps[-1000:]
    else:
        data_labels = ts_df.iloc[:, 3].values.astype(np.int)
    return data_vals, data_timestamps, data_labels


# plot
def plot_figure(test_vals, predictions, sensorname, model_name='ARIMA', test_labels=None, rate=0):
    figsize = (12, 7)
    plt.figure(figsize=figsize)
    plt.plot(test_vals, label='Actuals')
    plt.plot(predictions, color='red', label='Predicted')
    plt.legend(loc='upper left')
    plt.title("{0} for Crackmeter-2020-{1}-r{2}".format(model_name, sensorname, int(rate * 100)))
    # 保存图片
    plt.savefig("figure/{0}_Crackmeter-2020-{1}-r{2}.png".format(model_name, sensorname, int(rate * 100)))
    plt.show()


# 保存预测数据
def save_result(test_vals, predictions, sensorname, test_labels=None, data_timestamp=None, model_name='ARIMA', rate=0):
    predicted_df = pd.DataFrame()
    predicted_df['actuals'] = test_vals
    predicted_df['predicted'] = predictions
    predicted_df['actuals_labels'] = test_labels
    # predicted_df['timestamp'] = data_timestamp
    print(predicted_df.head())
    writer = pd.ExcelWriter("result/{0}_Crackmeter-2020-{1}-r{2}.xlsx".format(model_name, sensorname, int(rate * 100)),
                            encoding="utf-8-sig")
    predicted_df.to_excel(writer, "sheet1")
    writer.save()
    print("数据保存成功")


# 添加异常点的函数，从高斯函数中取样，添加至正常点上使其变成异常点
def add_abnormal(filename, rate=0.01, mu=1, sigma=0.1):
    abnormal_filename = filename.split('.')[0] + '_ab_r' + str(int(rate * 100)) + '.' + filename.split('.')[1]
    ts_df = pd.read_csv(filename, header=None)
    data_vals = ts_df.iloc[:, 2].values.astype(np.float)
    abnormal_num = int(data_vals.shape[0] * rate)
    abnormal_index = random.sample(range(0, data_vals.shape[0]), abnormal_num)
    label_vals = np.zeros(data_vals.shape[0], int)
    label_vals[abnormal_index] = 1
    abnormal_sample = np.array(
        [random.choice((-1, 1)) * float(sigma * np.random.rand(1) + mu) for i in range(abnormal_num)])
    data_vals[abnormal_index] += abnormal_sample
    ts_df.iloc[:, 2] = data_vals
    ts_df['3'] = label_vals
    ts_df.to_csv(abnormal_filename, index=None, header=None)
    print('-----Abnormal points have been added!-----')
    return abnormal_filename


def csv_tag(filename, rate=0.01):
    # abnormal_filename = filename.split('.')[0] + '_ab_r' + str(int(rate * 100)) + '.' + filename.split('.')[1]
    ts_df = pd.read_csv(filename, header=None)
    data_vals = ts_df.iloc[:, 2].values.astype(np.float)
    diff = np.diff(data_vals, n=1)
    threshold = np.percentile(diff, (1 - rate) * 100)
    label_vals = np.zeros(data_vals.shape[0], int)
    diff = np.insert(diff, 0, 0)
    abnormal_index = (diff >= threshold)
    label_vals[abnormal_index] = 1
    ts_df['3'] = label_vals
    print(ts_df[ts_df['3'] > 0])
    ts_df.to_csv(filename, index=None, header=None)
    print('-----Abnormal points have been tagged!-----')
    return


def excel_tag(filename, rate):
    ts_df = pd.read_excel(filename, sheet_name='sheet1')
    data_vals = ts_df.actuals.values.astype(np.float)
    diff = np.diff(data_vals, n=1)
    threshold = np.percentile(diff, (1 - rate) * 100)
    label_vals = np.zeros(data_vals.shape[0], int)
    diff = np.insert(diff, 0, 0)
    abnormal_index = (diff >= threshold)
    label_vals[abnormal_index] = 1
    ts_df['actuals_labels'] = label_vals
    print(ts_df.head())
    writer = pd.ExcelWriter(filename,
                            encoding="utf-8-sig")
    ts_df.to_excel(writer, "sheet1")
    writer.save()
    print('-----Abnormal points have been tagged!-----')
    return


def read_GNSS(filename, is_train=True):
    f = open(filename, "rb")
    a = pickle.load(f)
    a = a.T
    values1, values2, values3 = a[0], a[1], a[2]
    if is_train:
        values1, values2, values3 = values1[-1000:], values2[-1000:], values3[-1000:]
    f.close()
    return values1, values2, values3


def read_GNSSlabel(filename):
    f = open(filename, "rb")
    a = pickle.load(f)
    f.close()
    return a
