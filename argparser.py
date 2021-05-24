# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:08:09 2021

@author: bchan
"""

import argparse
from datetime import date

today = date.today()

parser = argparse.ArgumentParser(description='Parameters for train.py')

parser.add_argument('features', 
                    nargs='+',
                    help="Feature names to be used in training")

parser.add_argument('-n', '--name',
                    help="Features string to append to end of model name")

parser.add_argument('-r', '--rnd_num',
                    default=int(today.strftime('%m%d%y')),
                    type=int,
                    help="Rnd seed to initialize the model [Default: Today's date]")

parser.add_argument('-f', '--n_filters',
                    default=10,
                    type=int,
                    help="Number of convolutional filters in the first layer [Default: 10]")

parser.add_argument('-e', '--epochs',
                    default=1000,
                    type=int,
                    help="Number of epochs to train for [Default: 1000]")

parser.add_argument('-s', '--batch_size',
                    default=5,
                    type=int,
                    help="Batch size [Default: 5]")

parser.add_argument('-b', '--n_branches', 
                    default=1,
                    type=int,
                    help="Number of branches [Default: 1]")

parser.add_argument('--n_gpus', 
                    default=1,
                    type=int,
                    help="Number of GPUs to train model [Default: 1]")

parser.add_argument('--mem_frac',
                    default=0.9,
                    help="Fraction of GPU to use [Default: 0.9]")

parser.add_argument('--use_generator',
                    default=False,
                    help="Use data generator [Default: False]")

parser.add_argument('--input_size',
                    default=80,
                    type=int,
                    help="Length of the side of a training cube (pixels) [Default: 80]")

parser.add_argument('--val_split',
                    default=0.2,
                    help="Splits the last <x> [Default: 0.2]")

parser.add_argument('--patience_training',
                    default=100,
                    type=int,
                    help="Number of epochs without val_loss improvement before stopping [Default: 100]")

parser.add_argument('--total_samples',
                    default=1080,
                    type=int,
                    help="Number of samples for data generator [Default: 1080]")

parser.add_argument('--use_customloss',
                    default=False,
                    help="Use custom loss function [Default: False]")


args = parser.parse_args()



