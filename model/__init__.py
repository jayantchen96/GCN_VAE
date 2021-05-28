from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


# 图卷积层
class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, seq=1,
                 batch_size=None,
                 dropout=0.,
                 bias=False,
                 activation=F.relu):

        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(seq, input_dim, output_dim))
        self.bias = None
        if batch_size:
            self.weight = nn.Parameter(torch.randn(batch_size, seq, input_dim, output_dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(batch_size, seq, output_dim))
        else:
            if bias:
                self.bias = nn.Parameter(torch.zeros(seq, output_dim))

    def forward(self, inputs):  # (Batch_size, seq_len, num_sensor, num_features) , (num_sensors, num_sensors)
        # print('inputs:', inputs)
        x, x_adj = inputs

        x = F.dropout(x, self.dropout)

        # convolve
        xw = torch.matmul(x, self.weight)

        # print(xw.shape)
        # print(x_adj.shape)
        out = torch.matmul(x_adj, xw)
        # print(out.shape)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), x_adj


# 2层图卷积
class _GCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, seq, batch_size, dropout=0.5):
        super(_GCN, self).__init__()

        self.gcns = nn.Sequential(OrderedDict([
            ('gcn1', GraphConvolution(input_dim, hidden_dim, seq,
                                      batch_size=batch_size,
                                      activation=F.relu,
                                      dropout=dropout)),

            ('gcn2', GraphConvolution(hidden_dim, output_dim, seq,
                                      batch_size=batch_size,
                                      activation=F.relu,
                                      dropout=dropout))
        ]))

    def forward(self, inputs):
        x, x_adj = inputs
        x_adj = x_adj[0]
        x = self.gcns((x, x_adj))

        return x


if __name__ == '__main__':
    B = 1
    seq_len = 1000
    n = 20
    m = 15

    layer = GraphConvolution(input_dim=m,
                             output_dim=m * 2,
                             seq=seq_len,
                             batch_size=B,
                             dropout=0.5,
                             activation=F.relu)

    x = torch.randn(B, seq_len, n, m)  # (Batch_size, seq_len, num_sensor, num_features)
    adj = torch.randn(n, n)
    y, y_adj = layer((x, adj))
    print(y.shape)
