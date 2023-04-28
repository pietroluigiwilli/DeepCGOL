### Data functions:  This document contains the functions necessary to generate random frames.
### The Data class contains methods for generating a dataset made up of pairs of images, an evolved frame and an random unevolved frame.
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import torch
import torchvision
from tqdm import tqdm

### consider using pre-evolution

class Data():
  def __init__(self, frame_shape, steps, num_frames):
    super(Data, self).__init__()
    self.frame_shape = frame_shape
    self.rows=self.frame_shape[0]
    self.cols=self.frame_shape[1]
    self.steps = steps
    self.num_frames = num_frames

  def spawn(self):
    #population = np.random.randint(low=10, high=60, size=1)
    frame = np.zeros(self.frame_shape)#[rows, cols]
    population = int((self.frame_shape[0]*self.frame_shape[1])*0.15)
    lower_center = 0
    upper_center = int((frame.shape[1]))
    latitude = np.random.randint(low=lower_center, high=upper_center, size=population)
    longitude = np.random.randint(low=lower_center, high=upper_center, size=population)
    source = np.ones(population, dtype=int)
    frame[(longitude),(latitude)] = source
    return frame.astype(int)

  def torus(self, X):
    return self.reshape_input(np.array([
        np.pad(x.reshape(self.frame_shape), (self.steps,self.steps), mode = 'wrap')
        for x in X
    ]))

  def step(self, frame):
    frame = frame.astype(int)
    for i in range(self.steps):
        frame = evolve(frame)
    return frame.astype(int)

  def collection(self):
    return np.array([
        self.spawn()
        for _ in tqdm(range(self.num_frames))
    ]).astype(int)

  def snapshots(self,frame):
    #frame = cp_to_np(frame)
    if len(frame.shape) == 2:
      f = frame
    else:
      index = np.random.randint(0,frame.shape[0],1)
      f = frame[index]
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(6,6))
    ax1.imshow(f, cmap='viridis', interpolation='nearest')
    #plt.show()
    ax2.imshow(self.step(f), cmap='viridis', interpolation='nearest')

  def reshape_input(self, f):
    return f.reshape(f.shape[0], 1, f.shape[1], f.shape[2])

  def generator(self):
    X = self.collection()
    X_copy = np.copy(X)
    y = np.array([
        self.step(X_copy[f]) for f in tqdm(range(self.num_frames))
    ])

    X = self.reshape_input(X)
    y = self.reshape_input(y)
    print("Dataset generation successful!")
    return self.torus(X), y

def evolve(frame):
    neighbours = sum(np.roll(np.roll(frame, row, 0), col, 1)
                 for row in (-1, 0, 1) for col in (-1, 0, 1)
                 if (row != 0 or col != 0))
    return (neighbours == 3) | (frame & (neighbours == 2)).astype(int)

def spawner(rows=20,cols=20):
    #population = np.random.randint(low=10, high=60, size=1)
    frame = np.zeros([rows,cols])#[rows, cols]
    population = int((rows*cols)*0.15)
    lower_center = 0
    upper_center = int((frame.shape[1]))
    latitude = np.random.randint(low=lower_center, high=upper_center, size=population)
    longitude = np.random.randint(low=lower_center, high=upper_center, size=population)
    source = np.ones(population, dtype=int)
    frame[(longitude),(latitude)] = source
    return frame.astype(int)
