import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
from data import load_data, preprocess_features, preprocess_adj
from model import Generator, Discriminator
from config import args
from utils import masked_loss, masked_acc

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)

# load data

device = torch.device('cpu')
# Batch_size, Sequence/Time, Location, Features
x_train = torch.randn(1, 128, 8, 8, device=device)
x_train_exter = torch.randn(1, 128, 8, 6, device=device)
x_tr_adj = torch.randint(low=0, high=2, size=(1, 1, 8, 8), device=device).float()

netG = Generator(input_shape=(1, 128, 8, 8), gru_dim=60, num_features_exter=6)
netE = Discriminator()
netG.to(device)
optimizer = optim.Adam(netG.parameters(), lr=args.learning_rate)

netG.train()
for epoch in range(args.epochs):

    out = netG((x_train, x_tr_adj, x_train_exter))
    out = out[0]

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    if epoch % 10 == 0:
        print('Train: [{0}/{1}]\t'.format(epoch, args.epochs))
        # 'Loss: {0}\t'.format(loss.item()))

# net.eval()

# out = net((feature, support))
# out = out[0]
