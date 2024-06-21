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
    
    #Load params
    with open(loaddir + '/params.json', 'r') as fp:
        params = json.load(fp)
    params['hparams'] = hparams
    params['label'] = label
    
    #Load data
    data = datamodule(loaddir, params)

    #Model
    MLmodel = model(params)
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
                         callbacks=checkpoint_callback,
                         profiler='simple')#,
                         #use_distributed_sampler=False)
                         #barebones=True)
    trainer.fit(MLmodel, data)
    end = time.time()
    Ctime = end - start #computation time
    print("Training time: %fs"%Ctime)