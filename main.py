from config import get_cfg_defaults
from utils.training import Trainer
from utils.testing import Tester
from utils.make_dataset import MyDataset
from model.GCN_VAE import GCN_VAE
import torch
import os

cfg = get_cfg_defaults()

dataset_name = cfg.TRAIN.DATASET
max_epoch = cfg.TRAIN.MAX_EPOCH
use_gpu = cfg.TRAIN.USE_GPU
learning_rate = cfg.TRAIN.LEARNING_RATE
batch_size = cfg.TRAIN.BATCH_SIZE
valid_proportion = cfg.TRAIN.VALID_PORTION
seed_for_reproduce = cfg.TRAIN.SEED
model = None

time_window = cfg.MODEL.WINDOW
num_devices = cfg.MODEL.NUM_DEVICES
num_sensors = cfg.MODEL.NUM_SENSORS
gcn_hidden_dim = cfg.MODEL.GHD
gcn_out_dim = cfg.MODEL.GOD
gru_dim = cfg.MODEL.GROD
z_dim = cfg.MODEL.Z_DIM

if __name__ == '__main__':

    model = GCN_VAE(input_shape=(time_window, num_devices, num_sensors),
                    gcn_hidden_dim=gcn_hidden_dim,
                    gcn_out_dim=gcn_out_dim,
                    gru_dim=gru_dim,
                    z_dim=z_dim)

    save_path = os.path.join('./save', dataset_name, f'gcn_vae_{gcn_hidden_dim}_{gcn_out_dim}_{gru_dim}_{z_dim}.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, save_path, optimizer, max_epoch=max_epoch, use_gpu=use_gpu)

    train_set = MyDataset(dataset=dataset_name, time_windows=time_window, step=1, is_train=True)

    trainer.fit(train_set, valid_proportion=valid_proportion, batch_size=batch_size, random_state=seed_for_reproduce)

    # ===============================================================================================================

    tester = Tester(model, save_path, use_gpu=use_gpu)

    test_set = MyDataset(dataset=dataset_name, time_windows=time_window, step=1, is_train=False)

    tester.evaluate(test_set, batch_size=1, random_state=seed_for_reproduce)
