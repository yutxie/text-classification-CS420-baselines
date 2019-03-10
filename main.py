import os
import time
import argparse
import logging as log

import torch

import models

from train import train
from dataset import Task
from evaluate import evaluate

parser = argparse.ArgumentParser(description='Text Classification')

# environment
parser.add_argument('--device',         type=int,   default=-1,     help='gpu device id, -1 if cpu')
# log
parser.add_argument('--log_every',      type=int,   default=1,    help='log train how many every passes')
parser.add_argument('--eval_every',     type=int,   default=1,    help='evaluate how many every passes')
parser.add_argument('--save_every',     type=int,   default=1000,   help='save model how many every passes')
# train
parser.add_argument('--n_epochs',       type=int,   default=500,    help='how many epochs')
parser.add_argument('--batch_size',     type=int,   default=256,    help='how many instances in a batch')
parser.add_argument('--lr',             type=float, default=3e-3,   help='learning rate')
# data
parser.add_argument('--data_dir',       type=str,   default='data/')
parser.add_argument('--run_dir',        type=str,   default='run/')
# model
parser.add_argument('--model',          type=str,   default='BiLSTM')
parser.add_argument('--d_feature',      type=int,   default=300)
parser.add_argument('--d_hidden',       type=int,   default=300)
parser.add_argument('--n_layers',       type=int,   default=1)
parser.add_argument('--dropout',        type=float, default=.5)

args = parser.parse_args()

# make dirs
os.makedirs(args.run_dir, exist_ok=True)

# set log
log.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, 'log.txt'), mode='w'))
log.info(str(vars(args)))

# parse device
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

#########################################

if __name__ == "__main__":

    # dataset
    task = Task(args)

    # model
    Model = getattr(models, args.model)
    model = Model(args, task)

    # train
    train(args, model, task)
    evaluate(args, model, task)