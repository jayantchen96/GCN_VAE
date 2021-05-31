from __future__ import print_function

from yacs.config import CfgNode as CN
import os

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# Data path
# _C.SYSTEM.DATA_PATH = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'data')

_C.TRAIN = CN()
# dataset used
_C.TRAIN.DATASET = 'ghg'
# max epoch for model training
_C.TRAIN.MAX_EPOCH = 200
# whether use GPU or not
_C.TRAIN.USE_GPU = True
# learning rate
_C.TRAIN.LEARNING_RATE = 0.0001
# batch size
_C.TRAIN.BATCH_SIZE = 8
# validation proportion in training set
_C.TRAIN.VALID_PORTION = 0.1
# seed for reproduce
_C.TRAIN.SEED = 42

# model settings
_C.MODEL = CN()
# time window
_C.MODEL.WINDOW = 10
# num_devices
_C.MODEL.NUM_DEVICES = 7
# num_devices
_C.MODEL.NUM_FEATURES = 16
# GCN hidden dim
_C.MODEL.GHD = 32
# GCN output dim
_C.MODEL.GOD = 64
# GRU output dim
_C.MODEL.GROD = 32
# latent variable Z dim
_C.MODEL.Z_DIM = 32


def get_cfg_defaults():

    cfg = _C.clone()
    return cfg