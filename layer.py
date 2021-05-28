import torch
from torch import nn
from torch.nn import functional as F

from utils import sparse_dropout

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, seq=1,
                 batch_size=None,
                 num_features_nonzero=None,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero
        self.weight = nn.Parameter(torch.randn(seq, input_dim, output_dim))
        self.bias = None
        if batch_size:
            self.weight = nn.Parameter(torch.randn(batch_size, seq, input_dim, output_dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(batch_size, seq, output_dim))
        else:
            if bias:
                self.bias = nn.Parameter(torch.zeros(seq, output_dim))

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, x_adj = inputs

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.matmul(x, self.weight)
        else:
            xw = self.weight

        # print(xw.shape)
        # print(x_adj.shape)
        out = torch.matmul(x_adj, xw)
        # print(out.shape)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), x_adj
