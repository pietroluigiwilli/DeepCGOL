### Convulutional Neural Networks:  this script contains convolutional neural network architectures.
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import torch
import torchvision
from tqdm import tqdm


class ConvNetwork(torch.nn.Module):
  def __init__(self, Q, steps):
    super(ConvNetwork, self).__init__()

    self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=Q, kernel_size=(1+2*steps,1+2*steps), stride=1, padding=0, padding_mode="circular")
    self.conv2 = torch.nn.Conv2d(in_channels=Q, out_channels=1, kernel_size=(1+2*steps,1+2*steps), stride=1, padding=steps, padding_mode="circular")
    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout2d(p=0.2)
    self.prelu = torch.nn.PReLU()
    self.flatten = torch.nn.Flatten()
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    x = self.conv1(x)
    x = self.dropout(x)
    x = self.relu(x)
    x = self.conv2(x)
    y = self.sigmoid(x)
    return y
