import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import json
import time
from DataModule import *
from logger import *

def train(model, datamodule, hparams, loaddir, logdir, sublogdir, label):
    
    #Initialize logger and checkpointer
    logger, checkpoint_callback = initialize_logger(logdir, sublogdir, label)
    
    #Load data and params
    data = datamodule(loaddir, hparams)
    with open(loaddir + '/params.json', 'r') as fp:
        params = json.load(fp)
    params['hparams'] = hparams
    params['label'] = label

    #Model
    MLmodel = model(params)
    MLmodel = MLmodel.to(hparams['dtype'])
    
    #Training
    start = time.time() 
    trainer = pl.Trainer(logger=logger, 
                         accelerator=hparams['accelerator'], 
                         devices=hparams['devices'],
                         precision=hparams['precision'], 
                         max_epochs=hparams['max_epochs'], 
                         check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss', patience=hparams['early_stopping_patience'])])
    trainer.fit(MLmodel, data)
    end = time.time()
    Ctime = end - start #computation time
    print("Training time: %fs"%Ctime)