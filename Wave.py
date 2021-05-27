import numpy as np
import pandas as pd
import pickle as pkl
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def createSingleWave(timestamp, amplitude, phi, period, numOfperiod, type='sin'):
    '''
    timestamp: 时间戳个数
    amplitude: 振幅
    phi: 相位
    period: 周期
    numOfperiod: 周期数
    type: 波形
    '''
    x = np.linspace(0, numOfperiod * period, timestamp)
    if type == 'sin':
        y = amplitude * np.sin(2 * np.pi / period * (x + phi))
    else:
        y = amplitude * np.cos(2 * np.pi / period * (x + phi))
    return y.reshape(y.shape[0], 1, 1)


def createSensorWave(timestamp, numOfSens, numOfFeas):
    '''
    timestamp: 时间戳个数
    numOfSens: 传感器个数
    numOfFeas: 传感器特征数
    '''
    arr = None
    for i in range(numOfSens):
        # 每个传感器通过正余弦波形叠加随机生成波形
        y = i * createSingleWave(timestamp, 1, 0, 2 * np.pi, 20, type='sin') + (numOfSens - i) * createSingleWave(timestamp, 1, 0, 2 * np.pi, 20, type='cos')  #[950, 1, 1]
        
        # 同一个传感器的不同特征呈倍数关系
        features = []
        for j in range(1, numOfFeas + 1):
            features.append(y * j)
        ys = np.concatenate(tuple(features), axis=2)  
        if i == 0:
            arr = ys
        else:
            arr = np.concatenate((arr, ys), axis=1)
    return arr

# if __name__ == '__main__':
# arr = createSensorWave(timestamp=950, numOfSens=10, numOfFeas=3)
# print(arr.shape)
# with open('wave.pkl', 'wb') as f:
#     pkl.dump(arr, f)
