import random
import os
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def loss_function(x_hat, x, mu, log_var):
    batch_size = x.shape[0]
    reconstruction_loss = ((x_hat - x) ** 2).sum()

    mu = mu.reshape(batch_size, -1)
    log_var = log_var.reshape(batch_size, -1)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    loss = reconstruction_loss / (np.product(x.shape)) + kld_loss

    return loss, reconstruction_loss, kld_loss
