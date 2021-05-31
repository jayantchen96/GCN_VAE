import torch
import torch.nn as nn
from thop import profile, clever_format
from model import _GCN


class GCN_VAE(nn.Module):
    def __init__(self, input_shape, gcn_hidden_dim, gcn_out_dim, gru_dim, z_dim=32):
        super(GCN_VAE, self).__init__()

        batch_size, seq_len, num_sensors, num_feas = input_shape

        self.encoder = Encoder(input_shape=input_shape,
                               gcn_hidden_dim=gcn_hidden_dim,
                               gcn_out_dim=gcn_out_dim,
                               gru_dim=gru_dim,
                               z_dim=z_dim)

        self.decoder = Decoder(input_shape=(batch_size, seq_len, num_sensors, z_dim), output_dim=num_feas)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def loss_function(self, x_hat, x, mu, log_var):
        batch_size = x.shape[0]
        reconstruction_loss = ((x_hat - x) ** 2).sum()

        mu = mu.reshape(batch_size, -1)
        log_var = log_var.reshape(batch_size, -1)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = reconstruction_loss / batch_size + kld_loss

        return loss


class Encoder(nn.Module):
    def __init__(self, input_shape, gcn_hidden_dim, gcn_out_dim, gru_dim, z_dim, has_ext=False, ext_in_dim=None,
                 ext_gru_dim=None):
        super(Encoder, self).__init__()
        # 两层GCN
        assert len(input_shape) == 4
        batch_size, seq_len, num_sensors, num_features = input_shape

        self.num_sensors = num_sensors

        self.gcn = _GCN(input_dim=num_features,
                        output_dim=gcn_out_dim,
                        hidden_dim=gcn_hidden_dim,
                        seq=seq_len,
                        batch_size=batch_size)

        self.grus = [nn.GRU(input_size=gcn_out_dim,
                            hidden_size=gru_dim,
                            num_layers=1,
                            batch_first=True) for _ in range(num_sensors)]

        self.instance_norms = [nn.InstanceNorm1d(gru_dim) for _ in range(num_sensors)]

        self.has_ext = has_ext

        if has_ext:
            self.ext_gru = nn.GRU(input_size=ext_in_dim, hidden_size=ext_gru_dim, num_layers=2, batch_first=True)
            self.alpha = nn.Parameter(torch.randn(1), requires_grad=True)
            self.beta = nn.Parameter(torch.randn(1), requires_grad=True)

        # 均值 方差
        self.z_dim = z_dim
        self.mu_layers = [nn.Linear(gru_dim + z_dim, z_dim) for _ in range(num_sensors)]
        self.var_layers = [nn.Linear(gru_dim + z_dim, z_dim) for _ in range(num_sensors)]

    def forward(self, x, ext_x=None):
        x, x_adj = x  # (batch, seq, locations, features), (1, 1, features, features)

        # 进入GCN前先对邻接矩阵做变换  A' = D_-0.5 * (I + adj) * D_-0.5
        x_adj = torch.eye(x_adj.shape[0], dtype=x_adj.dtype, device=x_adj.device) + x_adj
        D = torch.zeros_like(x_adj)
        indexes = torch.arange(0, x_adj.shape[0])
        D[indexes, indexes] = torch.sum(x_adj, dim=1)
        D_inv_half = torch.sqrt(torch.inverse(D))
        x_adj = torch.matmul(torch.matmul(D_inv_half, x_adj), D_inv_half)

        x, _ = self.gcn((x, x_adj))

        x_list = []
        for i in range(self.num_sensors):
            x_temp, states_temp = self.grus[i](x[:, :, i, :])  # x_temp shape ==> (batch, seq_len, gru_dim)
            x_temp = self.instance_norms[i](x_temp)

            x_list.append(torch.unsqueeze(x_temp, dim=0))

        x = torch.cat(x_list, 0)

        if self.has_ext and ext_x is not None:
            ext_x = self.ext_gru(ext_x)
            x = self.alpha * x + self.beta * ext_x

        h = x.permute(1, 2, 0, 3)  # (batch_size, seq_len, num_sensors, gru_dim)

        all_z = []
        all_mu = []
        all_log_var = []
        for i in range(self.num_sensors):
            z_0 = torch.zeros((h.shape[0], self.z_dim), device=h.device)
            z_list = [z_0]
            mu_i_list = []
            log_var_i_list = []
            hh = h[:, :, i]  # (batch_size, seq_len, gru_dim)
            for t in range(hh.shape[1]):
                z_h_cat = torch.cat([z_list[-1], hh[:, t]], dim=1)
                mu_t = self.mu_layers[i](z_h_cat)
                log_var_t = self.var_layers[i](z_h_cat)
                std_t = torch.exp(0.5 * log_var_t)
                eps = torch.randn_like(mu_t, device=mu_t.device)
                z_t = mu_t + std_t * eps
                z_list.append(z_t)
                mu_i_list.append(mu_t)
                log_var_i_list.append(log_var_t)

            z_i = torch.stack(z_list[1:], dim=1)  # (batch_size, seq_len, z_dim)
            mu_i = torch.stack(mu_i_list, dim=1)
            log_var_i = torch.stack(log_var_i_list, dim=1)

            all_z.append(z_i)
            all_mu.append(mu_i)
            all_log_var.append(log_var_i)

        all_z = torch.stack(all_z, dim=2)  # (batch_size, seq_len, num_sensors, z_dim)
        all_mu = torch.stack(all_mu, dim=2)
        all_log_var = torch.stack(all_log_var, dim=2)

        return all_z, all_mu, all_log_var


class Decoder(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(Decoder, self).__init__()

        assert len(input_shape) == 4

        batch_size, seq_len, num_sensors, z_dim = input_shape

        self.grus = [nn.GRU(input_size=z_dim,
                            hidden_size=output_dim,
                            num_layers=1,
                            batch_first=True) for _ in range(num_sensors)]

        self.instance_norms = [nn.InstanceNorm1d(output_dim) for _ in range(num_sensors)]

        self.num_sensors = num_sensors

    def forward(self, x):
        x_list = []
        for i in range(self.num_sensors):
            x_temp, states_temp = self.grus[i](x[:, :, i, :])  # x_temp shape ==> (batch, seq_len, output_dim)
            x_temp = self.instance_norms[i](x_temp)

            x_list.append(x_temp)

        x = torch.stack(x_list, dim=2)  # (batch_size, seq_len, num_sensors, num_features)

        return x


if __name__ == '__main__':
    batch_size = 1
    seq_len = 10
    num_sensors = 7
    num_feas = 16

    # layer = Encoder((batch_size, seq_len, num_sensors, num_feas), gcn_hidden_dim=32, gcn_out_dim=64, gru_dim=128, z_dim=128)

    x = torch.randn(batch_size, seq_len, num_sensors, num_feas)  # (Batch_size, seq_len, num_sensor, num_features)
    adj = torch.randn(num_sensors, num_sensors)
    # y = layer((x, adj))

    print('VAE输入形状:', tuple(x.shape))
    # print('Encoder输出形状:', tuple(y.shape))

    model = GCN_VAE((batch_size, seq_len, num_sensors, num_feas), gcn_hidden_dim=32, gcn_out_dim=64, gru_dim=32,
                    z_dim=32)
    x_hat, mu, log_var = model((x, adj))
    print('VAE输出形状:', tuple(x_hat.shape))
    print('VAE输出mu:', tuple(mu.shape))
    print('VAE输出logvar:', tuple(log_var.shape))

    loss = GCN_VAE.loss_function(GCN_VAE, x_hat, x, mu, log_var)
    print(loss.item())

    # macs, params = profile(model, inputs=(x, adj))
    #
    # print(f'FLOPs: {macs * 2 / 1e9 : .3f}G')
    # print(f'Params: {params / 1e6 : .3f}M')

