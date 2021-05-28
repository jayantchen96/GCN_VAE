import os
import pickle
import random

import math
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import args
from layer import GraphConvolution
from utils import PredictorDataset
from utils import weights_init
from collections import OrderedDict


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class Generator(nn.Module):
    def __init__(self, input_shape, gru_dim, num_features_exter, gcn_dim=8):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_shape, gru_dim, num_features_exter, gcn_dim=gcn_dim)
        # 拼接的话，输入size就是gru_dim*2,主要因素和外部因素的hidden_size我都设置为一样了，直接double就行
        # 相加的话，就只是gru_dim
        self.decoder = Decoder(gru_dim, input_shape)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        del z
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, gcn_dim, gru_dim):
        super(Encoder, self).__init__()
        # 两层GCN
        assert len(input_shape) == 4
        batch_size, seq_len, num_sensors, num_features = input_shape

        self.gcn = _GCN(input_dim=num_features,
                        output_dim=gcn_dim,
                        seq=seq_len,
                        batch_size=batch_size)

        self.grus = nn.ModuleList(
            [nn.Sequential(nn.GRU(input_size=gcn_dim,
                                  hidden_size=gru_dim,
                                  num_layers=2,
                                  bias=False,
                                  batch_first=True),
                           nn.InstanceNorm1d(num_features=gru_dim))

             for _ in range(num_sensors)]
        )

    def forward(self, x):
        x, x_adj = x
        # (batch, seq, locations, features), (1, 1, features, features)
        # GCN处理
        x, _ = self.gcn((x, x_adj))
        # print(x.shape)

        # 主要因素GRU
        x_list = []
        # states_list = []
        for i in range(self.locations):
            # x[:, :, i, :] shape ==> (batch, seq_len, num_fea)
            # x_temp shape ==> (batch, seq_len, gru_dim)
            x_temp, states_temp = self.grus[i](x[:, :, i, :])
            x_list.append(torch.unsqueeze(x_temp, dim=0))
            # states_list.append(states_temp.view(1, states_temp.shape[0], states_temp.shape[1], states_temp.shape[2]))
        x = torch.cat(x_list, 0)

        return x


class Decoder(nn.Module):
    def __init__(self, x_dim, input_shape):
        super(Decoder, self).__init__()
        # 两层GRU
        self.locations = input_shape[-2]
        # 恢复成原来的特征数
        output_dim = input_shape[-1]
        # 这里的hidden states维度必须和输入的状态维度一致，因为是用输入的状态初始化的
        self.grus = nn.ModuleList(
            [nn.ModuleList([nn.GRU(input_size=x_dim, hidden_size=x_dim, num_layers=1, bias=False, batch_first=True),
                            nn.GRU(input_size=x_dim, hidden_size=x_dim, num_layers=1, bias=False, batch_first=True)])
             for i in range(self.locations)]
        )
        self.IN1 = nn.InstanceNorm1d(num_features=x_dim)
        self.linear1 = nn.Linear(in_features=x_dim * self.locations, out_features=output_dim * self.locations)
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x, states = x
        x_list = []
        for i in range(self.locations):
            x0, _ = self.grus[i][0](x[i, :, :, :])
            x0, _ = self.grus[i][1](x0)
            # x0 = x0[:,-1,:]
            # x_list.append(x0.view(1,))
            # x0, states0 = self.grus[i][0](x[i, :, :, :], states[i, :, :, :])
            # x0, states0 = self.grus[i][1](x0, states0)
            x_list.append(x0.view(1, x0.shape[0], x0.shape[1], x0.shape[2]))
        # [n, batch, seq, m]
        x = torch.cat(x_list, 0)
        # [batch, seq, n, m]
        x = x.permute(1, 2, 0, 3)

        # x = torch.squeeze(x, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # print(x.shape)
        x = self.IN1(x)
        # [batch, seq, n, m]还是保留了seq_len这个维度，其实不必要了
        # x = torch.unsqueeze(x, dim=1)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = x.view(x.shape[0], x.shape[1], self.locations, -1)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape, gru_dim, gcn_dim=8):
        super(Discriminator, self).__init__()
        # 两层GCN
        self.gcn = _GCN(input_dim=input_shape[-1],
                        output_dim=gcn_dim,
                        seq=input_shape[-3],
                        batch_size=input_shape[0])
        gru_x_dim = gcn_dim
        self.locations = input_shape[-2]
        # 主要GRU
        # self.grus = nn.GRU(input_size=gru_x_dim, hidden_size=gru_dim, num_layers=1, bias=False, batch_first=True)
        self.grus = nn.ModuleList(
            [nn.GRU(input_size=gru_x_dim, hidden_size=gru_dim, num_layers=2, bias=False, batch_first=True)
             for i in range(self.locations)]
        )
        self.IN1 = nn.InstanceNorm1d(num_features=gru_dim * self.locations)
        # 输出层
        self.linear = nn.Linear(in_features=gru_dim * self.locations, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, x_adj = x
        # [batch, seq, locations, features] [1, 1, features, features]
        x, _ = self.gcn((x, x_adj))
        # x = x.view(x.size(0), x.size(1), -1)
        # x, _ = self.grus(x)
        x_list = []
        # states_list = []
        for i in range(self.locations):
            # [batch, seq, hidden], [num_layers, batch, hidden] main
            x_temp, states_temp = self.grus[i](x[:, :, i, :])
            x_list.append(self.IN1(x_temp).view(1, x_temp.shape[0], x_temp.shape[1], x_temp.shape[2]))
            # states_list.append(states_temp.view(1, states_temp.shape[0], states_temp.shape[1], states_temp.shape[2]))
        x = torch.cat(x_list, 0)
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.IN1(x)
        x = self.linear(x)
        y = self.sigmoid(x)
        return y


class _GCN(nn.Module):

    def __init__(self, input_dim, output_dim, seq, batch_size, num_features_nonzero=None):
        super(_GCN, self).__init__()

        self.gcns = nn.Sequential(OrderedDict([
            ('gcn1', GraphConvolution(input_dim, args.hidden, seq,
                                      batch_size=batch_size,
                                      num_features_nonzero=num_features_nonzero,
                                      activation=F.relu,
                                      dropout=args.dropout,
                                      is_sparse_inputs=False)),
            ('gcn2', GraphConvolution(args.hidden, output_dim, seq,
                                      batch_size=batch_size,
                                      num_features_nonzero=num_features_nonzero,
                                      activation=F.relu,
                                      dropout=args.dropout,
                                      is_sparse_inputs=False))
        ]))

    def forward(self, inputs):
        x, x_adj = inputs
        x_adj = x_adj[0]
        x = self.gcns((x, x_adj))

        return x


class GLGANTrainer:
    def __init__(self, data_name, input_shape, generator_path, discriminator_path, mode="Train"):
        self.data_name = data_name
        self.generator_path = generator_path
        self.discriminator_path = discriminator_path
        self.input_shape = input_shape
        self.batch_size = 1
        self.time_windows = 8
        self.ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
        self.device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.build_models(mode=mode)
        print('dataset name:{}'.format(data_name))
        print('time windows length:{}'.format(self.time_windows))

    def build_models(self, mode="Train"):
        if mode == "Train":
            # 初始化generator
            print("训练--创建初始化模型")
            self.netG = Generator(
                input_shape=(self.batch_size, self.time_windows, self.input_shape[0], self.input_shape[1]),
                gru_dim=10, num_features_exter=5, gcn_dim=self.input_shape[1]).to(self.device)
            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            self.netG.apply(weights_init)
            print(self.netG)
            # 初始化discriminator
            self.netD = Discriminator(
                input_shape=(self.batch_size, self.time_windows + 1, self.input_shape[0], self.input_shape[1]),
                gru_dim=10, gcn_dim=self.input_shape[1]).to(self.device)
            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
            self.netD.apply(weights_init)
            print(self.netD)
        else:
            print("测试---加载已保存模型")
            self.load()

    def train(self):
        train_data = PredictorDataset(data_name=self.data_name,
                                      time_windows=self.time_windows,
                                      step=1)
        # train_data = SinDataset(data_name=self.data_name,
        #                         time_windows=self.time_windows,
        #                         step=1)
        data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,
                                 shuffle=True)
        self.data_loader = data_loader
        lr = 0.00001
        beta1 = 0.5
        num_epochs = 200
        # f1_best = 0
        # threshold_best = 0
        # acc_best = 0
        # pre_best = 0
        # recall_best = 0
        # best_epoch = -1
        # 初始化loss函数及优化器
        criterion = nn.BCELoss().cuda(device=self.device)
        rmse_loss = RMSELoss().cuda(device=self.device)
        # nmse_loss = NMSELoss().cuda(device=self.device)
        optimizer_d = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_g = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        iteration_number = math.ceil(train_data.__len__() / self.batch_size)
        print('Generator save path:{}'.format(self.generator_path))
        # print('Discriminator save path:{}'.format(self.discriminator_path))
        print("Starting Training Loop...")

        # For each epoch
        for epoch in range(num_epochs):
            i = 0
            reconstruct_error_total = 0
            # nmse_error_total = 0
            g_losses_total = 0
            d_losses_total = 0
            for x_generator, target in data_loader:
                i += 1
                # generate noise label and decreasing noise input of discriminator over time
                x_generator = [x_generator[i].to(self.device) for i in range(len(x_generator))]
                # x_generator = x_generator.to(self.device)
                # target = [target[i].to(self.device) for i in range(len(target))]
                target = target.to(self.device)
                # 标签的维度:(batchsize,timestamp)
                label_true = torch.full((target.size()[0], target.size()[1], 1),
                                        random.uniform(0.9, 1.),
                                        device=self.device)
                label_fake = torch.full((target.size()[0], target.size()[1], 1),
                                        random.uniform(0., 0.1),
                                        device=self.device)
                # add noise
                # noise1 = torch.Tensor(x_true.size()).normal_(mean=0, std=0.1 * (num_epochs - epoch) / num_epochs).to(
                #     self.device)
                # noise2 = torch.Tensor(x_true.size()).normal_(mean=0, std=0.1 * (num_epochs - epoch) / num_epochs).to(
                #     self.device)

                # ====== update Discriminator first ======
                # generator forward
                self.netG.zero_grad()
                x_predict = self.netG(x_generator)

                # discriminator forward
                self.netD.zero_grad()
                output_true = self.netD((target, x_generator[1]))
                x_fake = x_generator[0]
                x_fake = torch.cat((x_fake, torch.unsqueeze(x_predict[:, -1, :, :], dim=1)), dim=1)
                output_fake = self.netD((x_fake, x_generator[1]))

                # computing loss、backward and update parameter
                # print(torch.min(output_true), torch.min(output_fake), torch.min(label_true), torch.min(label_fake))
                # print(torch.max(output_true), torch.max(output_fake), torch.max(label_true), torch.max(label_fake))
                loss_d = criterion(output_true, label_true) + criterion(output_fake, label_fake)
                loss_d.backward()
                optimizer_d.step()

                # ====== update Generator ======
                # generator forward
                self.netG.zero_grad()
                x_predict = self.netG(x_generator)

                # discriminator forward
                self.netD.zero_grad()
                output_true = self.netD((target, x_generator[1]))
                x_fake = x_generator[0]
                x_fake = torch.cat((x_fake, torch.unsqueeze(x_predict[:, -1, :, :], dim=1)), dim=1)
                output_fake = self.netD((x_fake, x_generator[1]))
                # x_true = target[0]
                # reconstruct_error = rmse_loss(x_predict, x_true[:, -1, :, :])
                # reconstruct_error = rmse_loss(x_predict[:, -1, :, :], target)
                reconstruct_error = rmse_loss(x_predict[:, -1, :, :], target[:, -1, :, :])

                # nmse_error = ((x_predict[:, -1, :, :]-target).pow(2)/(target*target+1e-8)).mean()/target.size(0)
                # print("均值：", nmse_error)
                # nmse_error = ((x_predict[:, -1, :, :] - target).pow(2) / (target * target + 1e-8)).sum() / target.size(0)
                # nmse_error = ((x_predict - target).pow(2) / (target * target + 1e-8)).sum() / target.size(0)
                # print("总和：", nmse_error)
                # computing loss、backward and update parameter
                loss_g = criterion(output_fake, label_true) + criterion(output_true, label_fake) + reconstruct_error
                loss_g.backward()
                # reconstruct_error.backward()
                optimizer_g.step()

                # Save Losses for plotting later
                g_losses_total = g_losses_total + loss_g.mean().item()
                d_losses_total = d_losses_total + loss_d.mean().item()
                reconstruct_error_total = reconstruct_error_total + reconstruct_error.item()
                # nmse_error_total = nmse_error_total + nmse_error.item()

                # if i%10 == 0:
                #     print("Batch: [{0}/{1}]\t gen_train_loss:{2:.2f}\tdis_train_loss:{3:.2f}".format(
                #         i, len(data_loader), g_losses_total, d_losses_total
                #     ))
                '''
                # 保存loss至tensorboard
                writer.add_scalar('encoder_generator_train_lose', loss_eg.item(), epoch * iteration_number + i)
                writer.add_scalar('discriminator_train_lose', loss_d.item(), epoch * iteration_number + i)
                writer.add_scalar('RMSE', RMSE, epoch * iteration_number + i)
                '''

            # if (epoch + 1) % 10 == 0:
            #     fixed_x = fixed_x.cuda(device=self.device)
            #     reconstruct_x = self.netG(self.netE(fixed_x)).view(-1)
            #     plot_data(x, reconstruct_x.cpu().detach().numpy(), epoch)
            # 每一个epoch都输出
            self.save_weight_EGD()
            # Output training stats
            # print(
            #     'Epoch: [%d/%d]\tLoss_D: %.4f\tLoss_EG: %.4f\tRMSE: %.4f'
            #     % (epoch + 1, num_epochs, d_losses_total / iteration_number, g_losses_total / iteration_number,
            #        reconstruct_error_total / iteration_number))
            print(
                'Epoch: [%d/%d]\tLoss_rmse: %.4f\tLoss_g: %.4f\tLoss_d: %.4f' % (epoch + 1, num_epochs,
                                                                                 reconstruct_error_total / iteration_number,
                                                                                 g_losses_total / iteration_number,
                                                                                 d_losses_total / iteration_number))
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                print("Target data:")
                print(target[0, -1, :, :].detach().cpu().numpy())
                print("Predict data:")
                print(x_predict[0, -1, :, :].detach().cpu().numpy())
            '''
            pre, acc, recall, f1, threshold = self.evaluate()
            # print('f1:{}'.format(f1))
            if f1 > f1_best:
                f1_best = f1
                threshold_best = threshold
                best_epoch = epoch
                pre_best = pre
                acc_best = acc
                recall_best = recall
                self.save_weight_EGD()
                print(
                    'precision:{}, recall:{}, current best f1 score:{}, threshold:{},current epoch:{}'.format(pre_best,
                                                                                                              recall_best,
                                                                                                              f1_best,
                                                                                                              threshold,
                                                                                                              epoch))

        print(
            'time windows length:{},best f1 score:{},best threshold:{},best epoch:{}'.format(self.time_windows, f1_best,
                                                                                             threshold_best,
                                                                                             best_epoch))
        with open('../experiment_data/Evaluate.txt', 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            f.write("timestamp: {0}\n".format(timestamp))
            f.write("数据集: {}\n".format(self.data_root))
            f.write('模型:{}\n'.format(self.discriminator_path))
            f.write("Accuracy: {0:.2%}".format(acc_best))
            f.write("Precision: {0:.2%}".format(pre_best))
            f.write("Recall: {0:.2%}".format(recall_best))
            f.write("F1-Score: {0:.2%}\n".format(f1_best))
            f.write("epoch: {}\n".format(best_epoch))
            f.write("threshold: {0:.2%}\n".format(threshold_best))
        '''

    def evaluate(self, dataname=None, setname=None, timestop=0):
        scores, labels = self.predict(dataname=dataname, setname=setname, timestop=timestop)
        # scores = []
        # labels = pickle.load(open('data/shanghai_test_label.pkl', 'rb'), encoding='utf-8')
        # labels = read_anomalies("data/labeled_anomalies.csv",dataname, setname, timestop)
        # labels = labels[self.time_windows:]
        labels = labels.reshape(-1)
        f1_best = 0
        threshold_best = 0
        pre_best = 0
        acc_best = 0
        recall_best = 0
        # print("正常点的Scores(20个): ", scores[labels == 0][:20])
        # print("异常点的Scores(20个): ", scores[labels == 1][:20])
        print("target labels(前50个): ", labels[:50])

        for threshold in np.arange(0., 1, 0.001):
            tmp_scores = scores.copy()
            tmp_scores[tmp_scores >= threshold] = 1
            tmp_scores[tmp_scores < threshold] = 0
            f1 = f1_score(labels, tmp_scores)
            if f1 > f1_best:
                print("predict labels(前50个):", tmp_scores[:50])
                f1_best = f1
                pre_best = precision_score(labels, tmp_scores)
                acc_best = accuracy_score(labels, tmp_scores)
                recall_best = recall_score(labels, tmp_scores)
                threshold_best = threshold

        return pre_best, acc_best, recall_best, f1_best, threshold_best

    def predict(self, scale=True, dataname=None, setname=None, timestop=0):
        # 加载标签
        # labels = pickle.load(open('data/shanghai_test_label.pkl', 'rb'), encoding='utf-8')
        # labels = read_anomalies("data/labeled_anomalies.csv", dataname, setname, timestop)
        labels = pickle.load(open('data/wave_label.pkl', 'rb'), encoding='utf-8')
        labels = labels[self.time_windows:]
        # len = labels.shape[1]
        # print(type(labels))
        predict_list = []
        count1 = 0
        count2 = 0
        with torch.no_grad():
            batch_size = self.batch_size
            # 预测的真实值也返回了
            test_data = PredictorDataset(data_name=self.data_name,
                                         time_windows=self.time_windows,
                                         step=1, mode='Test')
            test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

            scores = torch.zeros(size=(test_data.__len__(), self.input_shape[-2]), dtype=torch.float32,
                                 device=self.device)
            for i, data in enumerate(test_loader):
                # 将数据移到gpu上面
                x_generator = data[0]
                target = data[1].cuda(device=self.device)
                x_generator[0] = x_generator[0].cuda(device=self.device)
                x_generator[1] = x_generator[1].cuda(device=self.device)
                # x_generator[2] = x_generator[2].cuda(device=self.device)
                # 替换
                if i > 0:
                    for j in range(1, 9):
                        if i - j >= 0:
                            index = np.where(labels[i - j + 1] == 1)[0]
                            if len(index) != 0:
                                x_generator[0][:, -j, index, :] = predict_list[i - j][:, -1, index, :]
                x_predict = self.netG(x_generator)
                predict_list.append(x_predict)

                if 1 in labels[i] and count1 <= 100:
                    print("[异常点/{0}]:".format(i))
                    print("Target data:")
                    print(target[0, -1, :, :].cpu().numpy())
                    print("Predict data:")
                    print(x_predict[0, -1, :, :].cpu().numpy())
                    print("异常设备下标(从1开始)：", np.where(labels[i] == 1)[0] + 1)
                    count1 += 1
                else:
                    if count1 <= 100:
                        print("[非异常点/{0}]:".format(i))
                        print("Target data:")
                        print(target[0, -1, :, :].cpu().numpy())
                        print("Predict data:")
                        print(x_predict[0, -1, :, :].cpu().numpy())
                        count1 += 1

                # if 1 in labels[2*i+1] and count1 < 5:
                #     print("[异常点/{0}/{1}]:".format(i, len(test_loader)))
                #     print("Target data:")
                #     print(target[1, -1, :, :].cpu().numpy())
                #     print("Predict data:")
                #     print(x_predict[1, -1, :, :].cpu().numpy())
                #     count1 += 1

                # if 0 in labels[2*i+1] and count2 < 5:
                #     print("[非异常点/%d/%d]:" % (i, len(test_loader)))
                #     print("Target data:")
                #     print(target[1, -1, :, :].cpu().numpy())
                #     print("Predict data:")
                #     print(x_predict[1, -1, :, :].cpu().numpy())
                #     count2 += 1
                # if i%200 == 0:
                #     print("[%d/%d]:"%(i, len(test_loader)))
                #     print("Target data:")
                #     print(target[0].cpu().numpy())
                #     print("Predict data:")
                #     print(x_predict[0,-1,:,:].cpu().numpy())
                # print(x_predict.view(x_predict[:,-1,:,:].size(0), -1).shape)
                # print(target.view(target.size(0), -1).shape)
                error = torch.sqrt(torch.pow((x_predict[:, -1, :, :] - target[:, -1, :, :]), 2))
                error = torch.max(error, dim=-1)[0]
                # print(error.size())
                # 计算总得分
                scores[i * batch_size: i * batch_size + error.size(0)] = error

            # Scale error vector between [0, 1]
            if scale:
                scores = (scores - torch.min(scores)) / (torch.max(scores) - torch.min(scores))

            scores = scores.cpu().numpy()
            scores = scores.reshape(-1)
            # print("Scores: ",scores[:50])

            return scores, labels

    def print_genlabels(self):
        train_data = PredictorDataset(data_name=self.data_name,
                                      time_windows=self.time_windows,
                                      step=1)
        data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size,
                                 shuffle=True)
        self.data_loader = data_loader
        with torch.no_grad():
            for x_generator, target in data_loader:
                x_generator = [x_generator[i].to(self.device) for i in range(len(x_generator))]
                target = target.to(self.device)
                x_predict = self.netG(x_generator)
                output_true = self.netD((target, x_generator[1]))
                x_fake = x_generator[0]
                x_fake = torch.cat((x_fake, torch.unsqueeze(x_predict[:, -1, :, :], dim=1)), dim=1)
                output_fake = self.netD((x_fake, x_generator[1]))
                print("output_true: ", torch.squeeze(output_true).cpu().numpy())
                print("output_fake: ", torch.squeeze(output_fake).cpu().numpy())

    def save_weight_EGD(self):
        save_dir = "../model_parameter"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.netD, self.discriminator_path)
        torch.save(self.netG, self.generator_path)

    def load(self):
        self.netG = torch.load(self.generator_path)
        self.netD = torch.load(self.discriminator_path)
        # pretrained_dict = self.netG.state_dict()
        # pretrained_dict['encoder.gcn.layers.0.weight'] = pretrained_dict['encoder.gcn.layers.0.weight'][-1:,:,:,:]
        # pretrained_dict['encoder.gcn.layers.1.weight'] = pretrained_dict['encoder.gcn.layers.1.weight'][-1:, :, :, :]
        # self.netG.load_state_dict(pretrained_dict)
        # with torch.no_grad():
        #     for name, parameter in self.netG.named_parameters():
        #         if "encoder.gcn.layers.0.weight" in name:
        #             temp = parameter[-1:, :, :, :].clone()
        #             parameter.copy_(temp)
        #         if "encoder.gcn.layers.1.weight" in name:
        #             temp = parameter[-1:, :, :, :].clone()
        #             parameter.copy_(temp)
        #         print(name, ":", parameter.size())


data_name = 'sin'
# data_root = '../data/NASA/MSL/MSL_train.pkl'  # Root directory for dataset
timestamp = "20210512-232353"
# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
generator_path = '../model_parameter/Generator_model_' + data_name + '_' + timestamp + '.pt'
discriminator_path = '../model_parameter/Discriminator_model_' + data_name + '_' + timestamp + '.pt'

# create model A-(9,25) D-(312,12,25) E-(2769,13,25) G-(2446,6,25) shanghai-(10, 3)
model = GLGANTrainer(data_name=data_name,
                     input_shape=(10, 3),
                     generator_path=generator_path,
                     discriminator_path=discriminator_path,
                     mode="Test")

# train model
# model.train()
# model.print_genlabels()
# test model
# read_anomalies("data/labeled_anomalies.csv",data_name.split("_")[1],
#                                                  data_name.split("_")[0].upper(),
#                                                  7406)
# pre, acc, recall, f1, threshold = model.evaluate(data_name.split("_")[1],
#                                                  data_name.split("_")[0].upper(),
#                                                  7406)
pre, acc, recall, f1, threshold = model.evaluate()
print('precision:{}, recall:{}, current best f1 score:{}, threshold:{}'.format(pre, recall, f1, threshold))
