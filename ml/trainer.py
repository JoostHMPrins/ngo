import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm, trange
from ipywidgets import IntProgress
import json
import time
from DataModule import *
from logger import *

def train(model, hparams, loaddir, label, sublogdir):
    
    #Initialize logger and checkpointer
    #logger, checkpoint_callback = initializelogger(label, sublogdir)
    
    #Load data and params
    data = DataModule(loaddir, hparams)
    with open(loaddir + '/params.json', 'r') as fp:
        params = json.load(fp)
    params['hparams'] = hparams
    params['label'] = label

    #Model
    MLmodel = model(params)
    MLmodel = MLmodel.to(hparams['dtype'])
    MLmodel.params = params
    
    #Training
    start = time.time() 
    trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=20)
    # trainer = pl.Trainer(logger=logger, 
    #                      gpus=hparams['gpus'],
    #                      precision=hparams['precision'], 
    #                      max_epochs=hparams['max_epochs'], 
    #                      check_val_every_n_epoch=1,
    #                      callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss', patience=hparams['early_stopping_patience'])])
    trainer.fit(MLmodel, data)
    end = time.time()
    Ctime = end - start #computation time
    print("Training time: %fs"%Ctime)