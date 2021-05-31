import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.metrics import compute_metrics
from utils import seed_everything


class Tester(object):

    def __init__(self, model, model_path, use_gpu=True):
        super(Tester, self).__init__()

        self._model = model
        self._model_path = model_path
        self._use_gpu = use_gpu
        self._load_model()

    def model(self):
        return self._model

    def _load_model(self):
        self._model.load_state_dict(torch.load(self._model_path))

    def evaluate(self, test_dataset, batch_size=1, random_state=42):
        seed_everything(random_state)
        print('testing size:', len(test_dataset))

        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

        device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")
        self._model.to(device)

        print("==> start testing ...")
        self._model.eval()

        rmse5_list = []
        mae5_list = []
        nmse5_list = []
        rmse10_list = []
        mae10_list = []
        nmse10_list = []
        rmse30_list = []
        mae30_list = []
        nmse30_list = []
        scaler = test_dataset.minmax_scaler
        for i, (x, adj_matrix, mask5, mask10, mask30) in enumerate(test_loader):
            x = x.type(torch.FloatTensor).to(device)
            adj_matrix = adj_matrix.type(torch.FloatTensor).to(device)
            mask5 = mask5.type(torch.FloatTensor).to(device)
            mask10 = mask10.type(torch.FloatTensor).to(device)
            mask30 = mask30.type(torch.FloatTensor).to(device)

            x_hat, _, _ = self._model((x, adj_matrix))

            assert x_hat.shape == x.shape == mask5.shape == mask10.shape == mask30.shape

            ori_shape = x_hat.shape
            x_hat = x_hat.reshape(-1, x_hat.shape[-1])
            x_hat = scaler.inverse_transform(x_hat).reshape(ori_shape)
            x = x.reshape(-1, x.shape[-1])
            x = scaler.inverse_transform(x).reshape(ori_shape)

            rmse5, mae5, nmse5 = compute_metrics(x_hat.cpu().data.numpy(), x.cpu().data.numpy(),
                                                 mask5.cpu().data.numpy())
            rmse10, mae10, nmse10 = compute_metrics(x_hat.cpu().data.numpy(), x.cpu().data.numpy(),
                                                    mask10.cpu().data.numpy())
            rmse30, mae30, nmse30 = compute_metrics(x_hat.cpu().data.numpy(), x.cpu().data.numpy(),
                                                    mask30.cpu().data.numpy())

            rmse5_list.append(rmse5)
            mae5_list.append(mae5)
            nmse5_list.append(nmse5)
            rmse10_list.append(rmse10)
            mae10_list.append(mae10)
            nmse10_list.append(nmse10)
            rmse30_list.append(rmse30)
            mae30_list.append(mae30)
            nmse30_list.append(nmse30)

        rmse5 = np.nanmean(rmse5_list)
        mae5 = np.nanmean(mae5_list)
        nmse5 = np.nanmean(nmse5_list)
        rmse10 = np.nanmean(rmse10_list)
        mae10 = np.nanmean(mae10_list)
        nmse10 = np.nanmean(nmse10_list)
        rmse30 = np.nanmean(rmse30_list)
        mae30 = np.nanmean(mae30_list)
        nmse30 = np.nanmean(nmse30_list)

        print("=" * 10 + " Testing Results " + "=" * 10)
        print(f'RMSE_5: {rmse5:.4f}, MAE_5: {mae5:.4f}, NMSE_5: {nmse5:.4f}')
        print(f'RMSE_10: {rmse10:.4f}, MAE_10: {mae10:.4f}, NMSE_10: {nmse10:.4f}')
        print(f'RMSE_30: {rmse30:.4f}, MAE_30: {mae30:.4f}, NMSE_30: {nmse30:.4f}')
