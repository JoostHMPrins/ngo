# Copyright 2025 Joost Prins

# Standard
import time

# 3rd Party
import pytorch_lightning as pl

# Local
import ngo.ml.logger as logger_module

def train(model, datamodule, hparams, logdir, sublogdir, label):
    """
    Train a machine learning model using PyTorch Lightning.

    Args:
        model (class): The machine learning model class.
        datamodule (class): The data module class for loading data.
        hparams (dict): Hyperparameters for training.
        logdir (str): Directory for tensorboard log files and checkpoints.
        sublogdir (str): Subdirectory for tensorboard log files and checkpoints.
        label (str): Label/name of the model.

    Returns:
        None
    """
    #Initialize logger and checkpointer
    logger, checkpoint_callback = logger_module.initialize_logger(logdir, sublogdir, label)
    #Load data
    data = datamodule(hparams)
    #Model
    MLmodel = model(hparams)
    #Training
    start = time.time() 
    trainer = pl.Trainer(logger=logger, 
                         accelerator=hparams['accelerator'], 
                         devices=hparams['devices'],
                        #  strategy="ddp",
                         precision=hparams['precision'], 
                         max_epochs=hparams['epochs'], 
                         check_val_every_n_epoch=1,
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=checkpoint_callback)
    trainer.fit(MLmodel, data)
    end = time.time()
    Ctime = end - start #computation time
    print("Training time: %fs"%Ctime)