from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


# 图卷积层
class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, seq=1,
                 dropout=0.,
                 bias=True,
                 activation=F.relu):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.has_bias = bias
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(seq, input_dim, output_dim))

        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):  # (Batch_size, seq_len, num_sensor, num_features) , (1, 1, num_sensors, num_sensors)
        # print('inputs:', inputs)
        x, x_adj = inputs

        x = F.dropout(x, self.dropout)

        # H' = act(adj * H * w)

        xw = torch.matmul(x, self.weight)
        x = torch.matmul(x_adj, xw)
        x = self.activation(x)

        if self.has_bias:
            x += self.bias

        return x, x_adj


# 2层图卷积
class _GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, seq, dropout=0.5):
        super(_GCN, self).__init__()

        self.gcns = nn.Sequential(OrderedDict([
            ('gcn1', GraphConvolution(input_dim, hidden_dim, seq, activation=F.relu, dropout=dropout)),
            ('gcn2', GraphConvolution(hidden_dim, output_dim, seq, activation=F.relu, dropout=dropout))
        ]))

    def forward(self, inputs):
        x, x_adj = inputs
        x, x_adj = self.gcns((x, x_adj))

        return x, x_adj


if __name__ == '__main__':
    x = torch.randn(8, 10, 7, 16)
    adj = torch.randn(7, 7)
    layer = _GCN(input_dim=16, output_dim=32, hidden_dim=20, seq=10)

    x_hat, adj = layer((x, adj))
    print(x_hat.shape)
    print(adj.shape)
