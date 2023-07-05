import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def initializelogger(label, logsubdir):
    
    logdir = '../logs'
    if not os.path.isdir(logdir + '/' + logsubdir):
        os.makedirs(logdir + '/' + logsubdir, exist_ok=True)
        
    version_name = label
    checkpoint_path = logdir + '/' + logsubdir + '/' + version_name

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, monitor='val_loss', mode='min', verbose=False, save_top_k=1)
    logger = TensorBoardLogger(logdir, name=logsubdir, version=label)

    return logger, checkpoint_callback