import torch.nn as nn



class CNN_BiLSTM_Encoder(nn.Module):
    def __init__(self, input_shape, gru_dim):
        super(CNN_BiLSTM_Encoder, self).__init__()
        batch_size, seq_len, num_sensors, num_features = input_shape
        self.cnn_1 = nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=(3, 1), padding=1)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), padding=1)
        self.gru = nn.GRU(input_size=num_sensors * num_features, hidden_size=gru_dim, num_layers=1, bias=False,
                          batch_first=True, bidirectional=True)
        self.IN = nn.InstanceNorm1d(num_features=gru_dim)

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x, states = self.gru(x)
        z = self.IN(x)
        return z


class CNN_BiLSTM_Decoder(nn.Module):
    def __init__(self, input_shape, gru_dim):
        super(CNN_BiLSTM_Decoder, self).__init__()
        batch_size, seq_len, self.num_sensors, self.num_features = input_shape
        trans_dim = self.num_sensors * self.num_features
        self.gru = nn.GRU(input_size=gru_dim, hidden_size=trans_dim, num_layers=1, bias=False,
                          batch_first=True, bidirectional=True)
        self.IN = nn.InstanceNorm1d(num_features=trans_dim)
        self.cnn_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 1), padding=1)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=self.num_features, kernel_size=(3, 1), padding=1)

    def forward(self, z):
        z, states = self.gru(z)
        z = self.IN(z)
        z = z.view(z.shape[0], z.shape[1], self.num_sensors, self.num_features)
        z = self.cnn_1(z)
        x = self.cnn_2(z)
        return x

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, gru_dim):
        super(CNN_BiLSTM, self).__init__()
        self.Encoder = CNN_BiLSTM_Encoder(input_shape, gru_dim)
        self.Decoder = CNN_BiLSTM_Decoder(input_shape, gru_dim)

    def forward(self, x):
        z = self.Encoder(x)
        x = self.Decoder(z)
        return x
