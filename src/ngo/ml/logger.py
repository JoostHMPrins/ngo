# Copyright 2025 Joost Prins

# Standard
import os

# 3rd party
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def initialize_logger(logdir, sublogdir, label):
    
    if not os.path.isdir(logdir + '/' + sublogdir):
        os.makedirs(logdir + '/' + sublogdir, exist_ok=True)
        
    checkpoint_path = logdir + '/' + sublogdir + '/' + label

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss', mode='min', verbose=False, every_n_epochs=1, save_top_k=1)
    logger = TensorBoardLogger(logdir, name=sublogdir, version=label)

    return logger, checkpoint_callback