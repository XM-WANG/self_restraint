import os
import sys
import time
import random
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size,*args,**kwargs):
    """ Load data and build dataloader.

    Parameters
    ----------
    batch_size : int
        batch size for batch training.

    Returns
    -------
    trainloader : Dataloader
        Dataloader for training.

    testloader : Dataloader
        Dataloader for test.

    validloader : Dataloader
        Dataloader for validation.
    """
    trainloader, testloader, validloader = None, None, None
    return trainloader, testloader, validloader

def setup_seed(seed):
    """ Setup random seed to avoid the impact of randomness.

    Parameters
    ----------
    seed : int
        random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logs(root_folder,arg_dict):
    """ Setup logger.

    Parameters
    ----------
    root_folder : str
        The root path of log file.
    arg_dict : dict
        The dictionary of hyper-parameters, like {lr: 0.01, bs: 256}.

    Returns
    -------
    logger : logger
        The logger.

    log_path : str
        The log file path end with .log

    exp_path : str
        The path for save trained model parameters end with .pkl
    """
    time_stamp = "-".join([str(x) for x in list(time.localtime(time.time()))])
    sub_folder = os.path.join(root_folder,time_stamp)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    args_stamp = "_".join([f"{str(k)}_{str(v)}" for k,v in arg_dict.items()])
    log_name = args_stamp + ".log"
    exp_name = args_stamp + ".pkl"
    log_path = os.path.join(sub_folder,log_name)
    exp_path = os.path.join(sub_folder,exp_name)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level=logging.INFO)
    logger.addHandler(file_handler)
    return logger, log_path, exp_path
