import torch
import torch.nn as nn

from model import _GCN


class GCN_VAE(nn.Module):
    def __init__(self, input_shape, gcn_hidden_dim, gcn_out_dim, gru_dim):
        super(GCN_VAE, self).__init__()

        self.encoder = Encoder(input_shape=input_shape,
                               gcn_hidden_dim=gcn_hidden_dim,
                               gcn_out_dim=gcn_out_dim,
                               gru_dim=gru_dim)

        self.mu_layer = None
        self.var_layer = None

        self.decoder = None

    def reparameterize(self, mu, log_var):
        pass

    def forward(self, x):
        x = self.encoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_shape, gcn_hidden_dim, gcn_out_dim, gru_dim):
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
                            num_layers=2,
                            bias=False,
                            batch_first=True) for _ in range(num_sensors)]

        self.instance_norms = [nn.InstanceNorm1d(gru_dim) for _ in range(num_sensors)]

    def forward(self, x):
        x, x_adj = x  # (batch, seq, locations, features), (1, 1, features, features)

        x, _ = self.gcn((x, x_adj))

        x_list = []
        for i in range(self.num_sensors):
            x_temp, states_temp = self.grus[i](x[:, :, i, :])  # x_temp shape ==> (batch, seq_len, gru_dim)
            x_temp = self.instance_norms[i](x_temp)

            x_list.append(torch.unsqueeze(x_temp, dim=0))

        x = torch.cat(x_list, 0)

        return x.permute(1, 2, 0, 3)  # (batch_size, seq_len, num_sensors, gru_dim)


if __name__ == '__main__':
    B = 1
    seq_len = 1000
    n = 20
    m = 10

    layer = Encoder((B, seq_len, n, m), gcn_hidden_dim=32, gcn_out_dim=64, gru_dim=128)

    x = torch.randn(B, seq_len, n, m)  # (Batch_size, seq_len, num_sensor, num_features)
    adj = torch.randn(n, n)
    y = layer((x, adj))
    print('Encoder输入形状:', tuple(x.shape))
    print('Encoder输出形状:', tuple(y.shape))
