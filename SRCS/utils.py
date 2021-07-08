import argparse
import random
import os
import torch
import numpy as np
import psutil

def get_argument_parser():
    parser = argparse.ArgumentParser(description="SVD")
    # Other parameters
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--train_val_split', type=int, default=0.7, help='percentage of split data into train and validation ')
    parser.add_argument('--ptest', type=float, default=0.05, help='percentage of sampling data for test')
    parser.add_argument('--dataf', help='spatio-temporal data file in csv', required=True)
    parser.add_argument('--infof', help='station info file in csv', required=True)
    parser.add_argument('--dainf', help='station info file in csv', required=True)
    parser.add_argument('--meshf', help='mesh file in csv', required=True)
    parser.add_argument('--ldapsdir', help='mesh file in csv', required=True)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--net', default='', help="path to net (to continue training)")
    parser.add_argument('--outf', default='./DAOU', help='folder to output and model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument('--limit_max_eofs', type=int, default=50, help="Limit the maximum number of eofs's components")
    parser.add_argument("--min_n_eofs", type=int, default=5, help="Minimum number of eofs's components to use training")
    parser.add_argument("--exvar_thres", type=float, default=0.99, help="Threshold of cumulative explained variance")
    parser.add_argument("--var", type=str, default = 'T3H', help="variable name[T3H,REH]",required=True)
    parser.add_argument("--sdate", type=str, default = '2019010100', help="start date, Format = YYYYMMDDHH")
    parser.add_argument("--edate", type=str, default = '2020010100', help="end date, Format = YYYYMMDDHH")
    parser.add_argument("--img_sdate", type=str, default = '2019010100', help="end date, Format = YYYYMMDDHH")
    parser.add_argument("--img_edate", type=str, default = '2019010103', help="end date, Format = YYYYMMDDHH")
    parser.add_argument("--fmt", type=str, default = '%Y%m%d%H', help="date format")

    return parser


def rev_scaling(var, data, out):
    if var == "T3H":
        minval = np.min(data) - 3
        maxval = np.max(data) + 3
    elif var == "REH":
        minval = 0.
        maxval = 100.
    return out * ( maxval - minval ) + minval

def set_seed(value):
    print("Random Seed: ", value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)

def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError:
        pass

def check_memory(mesg):
    print("===== %s" %(mesg))
    ### General RAM Usage
    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    memory_usage_percent = memory_usage_dict['percent']
    ### RAM Usage
    ram_total = int(psutil.virtual_memory().total) / 1024 / 1024
    ram_usage = int(psutil.virtual_memory().total - psutil.virtual_memory().available) / 1024 / 1024
    print(f"RAM total: {ram_total: 9.3f} MB")
    print(f"RAM usage: {ram_usage: 9.3f} MB / {memory_usage_percent}%")
    print("="*20)
