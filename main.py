import os
import time
import argparse
import logging as log

import torch

import models

from train import train
from evaluate import evaluate
from datasets import TaskDataset

parser = argparse.ArgumentParser(description='Text Classification')

# environment
parser.add_argument('--device',         type=int,   default=-1,     help='gpu device id, -1 if cpu')
# log
parser.add_argument('--log_every',      type=int,   default=10,     help='log train how many every passes')
parser.add_argument('--eval_every',     type=int,   default=100,    help='evaluate how many every passes')
parser.add_argument('--save_every',     type=int,   default=100,    help='save model how many every passes')
# train
parser.add_argument('--n_epochs',       type=int,   default=100,    help='how many epochs')
parser.add_argument('--batch_size',     type=int,   default=64,     help='how many instances in a batch')
parser.add_argument('--lr',             type=float, default=1e-4,   help='learning rate')
# data
parser.add_argument('--data_dir',       type=str,   default='data/')
parser.add_argument('--run_dir',        type=str,   default='run/')
# model
parser.add_argument('--model',          type=str,   default='BiLSTM')
parser.add_argument('--d_feature',      type=int,   default=300)
parser.add_argument('--d_hidden',       type=int,   default=500)
parser.add_argument('--n_layers',       type=int,   default=1)

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
    dataset = TaskDataset(args)

    # model
    Model = getattr(models, args.model)
    model = Model(args)
    model.to(args.device)

    # train
    train(args, model, dataset)
    evaluate(args, model, dataset)