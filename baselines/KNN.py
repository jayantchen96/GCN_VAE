import numpy as np


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
