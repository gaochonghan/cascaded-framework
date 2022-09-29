import numpy as np
import torch
import torch.distributed as dist
import os
import time
import socket
import logging
import shutil
import warnings
import torch
import torchvision
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import argparse
from utils.config import MInformerConfig
from torch.utils.data import Sampler
from random import shuffle


class MyDefiniteSampler(Sampler):
    def __init__(self, indice, device=None):
        self.indice = indice
        self.device = device

    def __iter__(self):
        shuffle(self.indice)
        print("shuffling", self.indice[:200])
        if self.device is not None:
            tensor_indices = torch.tensor(self.indice).to(self.device)
            dist.broadcast(tensor_indices, 0)
            self.indice = tensor_indices.tolist()
        return iter(self.indice)

    def __len__(self):
        return len(self.indice)



def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.88 ** ((epoch-1) // 1))}
        if args.model == "SCINet":
            lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, rank=None, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.rank = rank
        self.logger = logger

    def __call__(self, val_loss, model, path):
        if self.early_stop:
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.rank is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.logger.info(f'R{self.rank} EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.rank is None:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                self.logger.info(f'R{self.rank} Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.rank is not None:
            torch.save(model.state_dict(), path + '/' + '{}_checkpoint.pth'.format(self.rank))
        else:
            torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


class ProbMask():
    def __init__(self, B, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, :].expand(B, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def broadcast_coalesced(src, tensors):
    list_tensors = [i for i in tensors]
    shapes = [i.shape for i in list_tensors]
    for i in range(len(list_tensors)):
        list_tensors[i] = list_tensors[i].flatten()

    sizes = [t.numel() for t in list_tensors]
    data = torch.cat(list_tensors)

    dist.broadcast(data, src)
    cur = 0
    for step, (shape, size) in enumerate(zip(shapes, sizes)):
        list_tensors[step] = data[cur:cur+size].reshape(shape)
        cur += size
    return list_tensors


def all_reduce_coalesced(tensors, im_group=None):
    list_tensors = [i for i in tensors]
    shapes = [i.shape for i in list_tensors]
    for i in range(len(list_tensors)):
        list_tensors[i] = list_tensors[i].flatten()

    sizes = [t.numel() for t in list_tensors]
    try:
        data = torch.cat(list_tensors)
    except RuntimeError:
        print(list_tensors, type(list_tensors))
        exit()
    if im_group is None:
        dist.all_reduce(data)
    else:
        dist.all_reduce(data, group=im_group)
    cur = 0
    for step, (shape, size) in enumerate(zip(shapes, sizes)):
        list_tensors[step] = data[cur:cur+size].reshape(shape)
        cur += size
    return list_tensors

def find_free_port():
    # import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def setup():
    # init config
    config = MInformerConfig()

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False

    # For slurm available
    if "SLURM_NPROCS" in os.environ:
        # acquire world size from slurm
        config.world_size = int(os.environ["SLURM_NPROCS"])
        config.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = os.path.join(config.dist_path, "dist_url." + jobid + ".txt")
        if config.dist_file is not None:
            config.dist_url = "file://{}.{}".format(os.path.realpath(config.dist_file), jobid)
        elif config.rank == 0:
            if config.dist_backend == 'nccl' and config.infi_band:
                # only NCCL backend supports inifiniband
                interface_str = 'ib{:d}'.format(config.infi_band_interface)
                print("Use infiniband support on interface " + interface_str + '.')
                os.environ['NCCL_SOCKET_IFNAME'] = interface_str
                os.environ['GLOO_SOCKET_IFNAME'] = interface_str
                ip_str = os.popen('ip addr show ' + interface_str).read()
                ip = ip_str.split("inet ")[1].split("/")[0]
            else:
                if config.world_size == 1:  # use only one node
                    ip = '127.0.0.1'
                else:
                    ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            config.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(config.dist_url)
        else:
            while not os.path.exists(hostfile):
                time.sleep(5)  # wait for the main process
            with open(hostfile, "r") as f:
                config.dist_url = f.read()
        os.environ["MASTER_ADDR"] = str(config.dist_url.lstrip("tcp://").split(":")[0])
        os.environ["MASTER_PORT"] = str(config.dist_url.lstrip("tcp://").split(":")[1])
        print("dist-url:{} at PROCID {} / {}".format(config.dist_url, config.rank, config.world_size))
        return config


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def broadcast_coalesced(src, tensors):
    list_tensors = [i for i in tensors]
    shapes = [i.shape for i in list_tensors]
    for i in range(len(list_tensors)):
        list_tensors[i] = list_tensors[i].flatten()

    sizes = [t.numel() for t in list_tensors]
    data = torch.cat(list_tensors)

    dist.broadcast(data, src)
    cur = 0
    for step, (shape, size) in enumerate(zip(shapes, sizes)):
        list_tensors[step] = data[cur:cur+size].reshape(shape)
        cur += size
    return list_tensors


def all_reduce_coalesced(tensors, im_group=None):
    list_tensors = [i for i in tensors]
    shapes = [i.shape for i in list_tensors]
    for i in range(len(list_tensors)):
        list_tensors[i] = list_tensors[i].flatten()

    sizes = [t.numel() for t in list_tensors]
    try:
        data = torch.cat(list_tensors)
    except RuntimeError:
        print(list_tensors, type(list_tensors))
        exit()
    if im_group is None:
        dist.all_reduce(data)
    else:
        dist.all_reduce(data, group=im_group)
    cur = 0
    for step, (shape, size) in enumerate(zip(shapes, sizes)):
        list_tensors[step] = data[cur:cur+size].reshape(shape)
        cur += size
    return list_tensors


class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(torch.nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
