import numpy as np
def sin_data(timeWinNUm):
    sample_number = 354
    a = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165])
    b = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

    x_train = 10 * np.sin(a * np.pi / 180)
    x_train = np.expand_dims(x_train, 0)
    x_train = x_train.reshape(1, -1, 1)
    x_train = np.repeat(x_train, timeWinNUm, axis=0)
    x_train = x_train.reshape(-1, 1)
    x_train = np.expand_dims(x_train, 0)
    x_train = np.repeat(x_train, sample_number, axis=0)
    x_train = np.expand_dims(x_train, 0)

    y_train = 10 * np.sin(b * np.pi / 180)
    y_train = np.expand_dims(y_train, 0)
    y_train = y_train.reshape(1, -1, 1)
    y_train = np.repeat(y_train, timeWinNUm, axis=0)
    y_train = y_train.reshape(-1, 1)
    y_train = np.expand_dims(y_train, 0)
    y_train = np.repeat(y_train, sample_number, axis=0)
    y_train = np.expand_dims(y_train, 0)
    return x_train, y_train

x_train, y_train = sin_data(8)
print(x_train.shape, y_train.shape)