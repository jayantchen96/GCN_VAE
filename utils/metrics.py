import numpy as np

def compute_metrics(y_hat, y, mask):
    assert y_hat.shape == y.shape and y_hat.shape == mask.shape
    # 计算missing point位置与真实值的差异
    rmse = np.sqrt(((y_hat - y) ** 2)[mask == 0].mean())
    mae = np.abs(y_hat - y)[mask == 0].mean()
    # 计算一个样本的
    y_hat = y_hat.reshape(len(y), -1)
    y = y.reshape(len(y), -1)
    mask = mask.reshape(len(y), -1)
    nmse = np.mean(np.sum(((y_hat - y) * (1 - mask)) ** 2, axis=1) / np.sum((y * (1 - mask)) ** 2, axis=1), axis=0)

    return rmse, mae, nmse

