import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import json
import time
# from DataModule import *

from logger import *

def train(model, datamodule, hparams, loaddir, logdir, sublogdir, label):
    
    #Initialize logger and checkpointer
    logger, checkpoint_callback = initialize_logger(logdir, sublogdir, label)
    
    #Load data
    data = datamodule(loaddir, hparams)

    #Model
    # hparams['used_device'] = hparams['train_device']
    MLmodel = model(hparams)
    MLmodel = MLmodel.to(hparams['dtype'])

    #Training
    start = time.time() 
    trainer = pl.Trainer(logger=logger, 
                         accelerator='gpu', 
                         devices=hparams['devices'],
                         strategy="ddp",
                         precision=hparams['precision'], 
                         max_epochs=hparams['epochs'], 
                         check_val_every_n_epoch=1,
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=checkpoint_callback)
    
    trainer.fit(MLmodel, data)
    end = time.time()
    Ctime = end - start #computation time
    print("Training time: %fs"%Ctime)