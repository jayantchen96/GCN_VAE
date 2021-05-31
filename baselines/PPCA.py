import os
from scipy.linalg import orth
import numpy as np

class PPCA(object):

    def __init__(self, d=2, tol=1e-5, itr=500):
        self.raw = None
        self.data = None  # components
        self.W = None  # model_params  D x d
        self.means = None
        self.stds = None
        self.eig_vals = None  # eigen values
        self.d = d
        self.tol = tol
        self.itr = itr

    def _standard(self, X):
        if self.means is None or self.stds is None:
            raise RuntimeError("Fit model first")

        return (X - self.means) / self.stds

    def _inverse_standard(self, X):
        if self.means is None or self.stds is None:
            raise RuntimeError("Perform standardized operations first")

        return X * self.stds + self.means

    def _transform(self, data=None):

        if self.W is None:
            raise RuntimeError('Fit the data model first.')
        if data is None:
            return np.dot(self.data, self.W)
        return np.dot(data, self.W)

    def _calc_var(self):
        if self.data is None:
            raise RuntimeError('Fit the data model first.')

        data = self.data.T

        # variance calc
        var = np.nanvar(data, axis=1)
        total_var = var.sum()
        self.var_exp = self.eig_vals.cumsum() / total_var

    def fit(self, data):
        """Fit the model with parameter d specifying the number of components and
        verbose printing convergence output if required.
        """

        self.raw = data  # N x D
        self.raw[np.isinf(self.raw)] = np.max(self.raw[np.isfinite(self.raw)])  # 无穷的值用最大值代替

        data = self.raw.copy()
        N = data.shape[0]
        D = data.shape[1]

        def compute_para(x, kind=None):
            if kind == 'mean':
                means = np.nanmean(x, axis=0)

                if np.isnan(means).any():
                    index = np.where(np.isnan(means))
                    for i in index:
                        means[i] = (means[i + 1] + means[i - 1]) / 2.
                    index = np.where(np.isnan(means))
                    for i in index:
                        means[i] = np.nanmean(x)

                return means
            else:
                stds = np.nanmean(x, axis=0)
                if np.isnan(stds).any():
                    index = np.where(np.isnan(stds))
                    for i in index:
                        stds[i] = (stds[i + 1] + stds[i - 1]) / 2.
                    index = np.where(np.isnan(stds))
                    for i in index:
                        stds[i] = np.nanstd(x)
                return stds

        self.means = compute_para(data, kind='mean')
        self.stds = compute_para(data, kind='std')

        data = self._standard(data)  # 先进性数据的标准化
        observed = ~np.isnan(data)
        missing = np.sum(~observed)  # 缺失值数量
        data[~observed] = 0  # 缺失部分补0

        # initial
        if self.W is None:
            W = np.random.randn(D, self.d)
        else:
            W = self.W

        WW = np.dot(W.T, W)
        X = np.dot(np.dot(data, W), np.linalg.inv(WW))
        recon = np.dot(X, W.T)  # N x D
        recon[~observed] = 0
        ss = np.sum((recon - data) ** 2) / (N * D - missing)  # sigma^2

        v0 = np.inf
        counter = 0

        while True:

            Sx = np.linalg.inv(np.eye(self.d) + WW / ss)

            # E-step
            ss0 = ss
            if missing > 0:
                proj = np.dot(X, W.T)  # N x D
                data[~observed] = proj[~observed]
            X = np.dot(np.dot(data, W), Sx) / ss

            # M-step   更新 W 跟 sigma^2
            XX = np.dot(X.T, X)
            W = np.dot(np.dot(data.T, X), np.linalg.pinv(XX + N * Sx))
            WW = np.dot(W.T, W)
            recon = np.dot(X, W.T)
            recon[~observed] = 0
            ss = (np.sum((recon - data) ** 2) + N * np.sum(WW * Sx) + missing * ss0) / (N * D)

            # calc diff for convergence
            det = np.log(np.linalg.det(Sx))
            if np.isinf(det):
                det = abs(np.linalg.slogdet(Sx)[1])

            v1 = N * (D * np.log(ss) + np.trace(Sx) - det) + np.trace(XX) - missing * np.log(ss0)

            diff = abs(v1 / v0 - 1)

            if (diff < self.tol) and (counter > self.itr):
                break

            counter += 1
            v0 = v1

        W = orth(W)
        vals, vecs = np.linalg.eig(np.cov(np.dot(data, W).T))
        order = np.flipud(np.argsort(vals))  # 按照特征值从大到小排
        vecs = vecs[:, order]
        vals = vals[order]

        W = np.dot(W, vecs)

        # attach objects to class
        self.W = W
        self.data = data
        self.eig_vals = vals
        self._calc_var()

    def imputation(self, data=None):
        x = np.dot(self._transform(data=data), self.W.T)
        return self._inverse_standard(x)

    def save(self, fpath):
        np.save(fpath, self.W)

    def load(self, fpath):
        assert os.path.isfile(fpath)
        self.W = np.load(fpath)