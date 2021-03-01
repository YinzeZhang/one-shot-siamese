import os
import sys
import random
import itertools
import shutil
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from tqdm import trange
from PIL import Image
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

np.random.seed(0)
random.seed(0)

back_dir = './data/processed/background/'
eval_dir = './data/processed/evaluation/'
write_dir = './data/changed/'

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# get list of all alphabets
background_alphabets = [os.path.join(back_dir, x) for x in next(os.walk(back_dir))[1]]
background_alphabets.sort()

# list of all drawers (1 to 20)
background_drawers = list(np.arange(1, 21))
print("There are {} alphabets.".format(len(background_alphabets)))

# from 40 alphabets, randomly select 30
train_alphabets = list(np.random.choice(background_alphabets, size=30, replace=False))
valid_alphabets = [x for x in background_alphabets if x not in train_alphabets]

train_alphabets.sort()
valid_alphabets.sort()

train_write = os.path.join(write_dir, 'train')
for alphabet in train_alphabets:
    # print(str(k) + ": " + alphabet)
    # k = k + 1
    train_write_1 = train_write + '/' + alphabet.split('/')[-1] + '_'
    for char in os.listdir(alphabet):
        train_write_2 = train_write_1 + char
        char_path = os.path.join(alphabet, char)
        os.makedirs(train_write_2)
        for drawer in os.listdir(char_path):
            drawer_path = os.path.join(char_path, drawer)
            shutil.copyfile(
                drawer_path, os.path.join(
                    train_write_2, drawer
                )
            )

valid_write = os.path.join(write_dir, 'valid')
for alphabet in valid_alphabets:
    valid_write_1 = valid_write + '/' + alphabet.split('/')[-1] + '_'
    for char in os.listdir(alphabet):
        valid_write_2 = valid_write_1 + char
        char_path = os.path.join(alphabet, char)
        os.makedirs(valid_write_2)
        for drawer in os.listdir(char_path):
            drawer_path = os.path.join(char_path, drawer)
            shutil.copyfile(
                drawer_path, os.path.join(
                    valid_write_2, drawer
                )
            )

# get list of alphabets
test_alphabets = [os.path.join(eval_dir, x) for x in next(os.walk(eval_dir))[1]]
test_alphabets.sort()
test_write = os.path.join(write_dir, 'test')

for alphabet in test_alphabets:
    test_write_1 = test_write + '/' + alphabet.split('/')[-1] + '_'
    for char in os.listdir(alphabet):
        test_write_2 = test_write_1 + char
        char_path = os.path.join(alphabet, char)
        os.makedirs(test_write_2)
        for drawer in os.listdir(char_path):
            drawer_path = os.path.join(char_path, drawer)
            shutil.copyfile(
                drawer_path, os.path.join(
                    test_write_2, drawer
                )
            )