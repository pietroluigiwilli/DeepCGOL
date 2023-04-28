### CONWAY'S GAME OF LIFE
### Welcome to Conway's Game of Life
### the 3 rules of the game are:
#1. Any live cell with two or three live neighbours survives.
#2. Any dead cell with three live neighbours becomes a live cell.
#3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
import time
import msvcrt

from functions.data import Data, evolve, spawner
#Data(frame_shape, steps, num_frames)
from functions.training import train_simple, accuracy
# train_simple(network, x, y, epochs, suffix, steps,lr=0.01, dev="cpu", rounding=False)
# accuracy(y, x, t,steps=1)
from functions.test import Test
# Test(model, X, Y, steps, frame_shape)
from models.convnet import ConvNetwork
# ConvNetwork(Q, steps)
device = torch.device("cpu")

small_data = Data(frame_shape=(20,20), steps=1, num_frames=300)
big_data = Data(frame_shape=(20,20), steps=1, num_frames=300000)
X, Y = small_data.generator()

'''
network1 = ConvNetwork(50, 1)
network1 = train_simple(network1, X, Y, 10, "ConvNet",1)
'''
pretrained_network = ConvNetwork(50, 1)
pretrained_network.load_state_dict(torch.load(f"C:/Users/Pietro Willi/Desktop/Python Project/MACHINE LEARNING/Jewel2/model_weights/GOL_params_ConvNet_1steps_colab.pth"))

test1 = Test(pretrained_network,X,Y,1,(20,20))
error = test1.animate(0.1)
plt.imshow(error)
