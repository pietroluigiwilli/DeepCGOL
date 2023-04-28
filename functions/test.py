### Test functions:  This document contains a class that can be used to visualize the performance of the model and its predictions.
### the Test class contains various methods: the compare method, the see_errors method and the animate method.
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
from functions.data import Data, evolve
import time
import msvcrt


class Test(Data):
  def __init__(self, model, X, Y, steps, frame_shape):
    #super(Test, self).__init__()
    if 'torch' in str(type(X)):
      X = X.numpy()
    if 'torch' in str(type(Y)):
      Y = Y.numpy()
    self.model = model.cpu()
    self.X = X[int(len(X)*(275/300)):]
    with torch.no_grad():
      pred = self.model(torch.tensor(self.X).float())
      pred = torch.round(pred)
    self.pred = pred.detach().numpy()
    self.steps = steps
    self.t = Y[int(len(Y)*(275/300)):]
    self.frame_shape = frame_shape

  def reshape_input(self, f):
    return f.reshape(f.shape[0], 1, f.shape[1], f.shape[2])

  def torus(self, X):
    return np.array([
        np.pad(x.reshape(self.frame_shape), (self.steps,self.steps), mode = 'wrap')
        for x in X
    ])

  def reshape_output(self, frame):
    # for step == 1 take away 0 and -1
    # for step == 2 take away 0, 1, -1 and -2
    frames = (frame).copy()
    frames = np.delete(frames,np.linspace(-self.steps, self.steps-1, self.steps*2,dtype=int), 2)
    frames = np.delete(frames,np.linspace(-self.steps, self.steps-1, self.steps*2,dtype=int), 3)
    frames = np.squeeze(frames.astype('int'))
    return frames

  def compare(self, frame_n):
    print((self.X).shape)
    self.X = self.reshape_output(self.X)
    print((self.X).shape)
    X = np.squeeze(self.X)
    pred = np.squeeze(self.pred)
    t = np.squeeze(self.t)
    fig, axs = plt.subplots(2, 2, figsize=(11, 11))
    axs[0,0].imshow(X[frame_n], cmap='viridis', interpolation='nearest')
    axs[0,0].set_title('Initial Frame')

    #plt.subplot(2, 2, 2)
    axs[1,0].imshow(t[frame_n], cmap='viridis', interpolation='nearest')
    axs[1,0].set_title('Test Frame (Computationally Evolved)')

    #plt.subplot(2, 2, 3)
    axs[1,1].imshow(pred[frame_n], cmap='viridis', interpolation='nearest')
    axs[1,1].set_title('Predicted Frame (CNN)')

    diff = np.subtract(t[frame_n], pred[frame_n])
    diff = np.absolute(diff)
    #plt.subplot(2, 2, 4)
    axs[0,1].imshow(diff, cmap='viridis', interpolation='nearest')
    axs[0,1].set_title('Difference')

    plt.show()
    perform = 100*(((t).shape[0]*(t).shape[1]*(self.t).shape[2])-np.sum(np.absolute(np.subtract((t), pred))))/((t).shape[0]*(t).shape[1]*(t).shape[2])
    percent_diff = 100*diff.sum()/(t)[frame_n].sum()
    print(f"Percentage Difference: {percent_diff}%\n")
    print(f"Overall Model Accuracy {perform} %")

  def see_errors(self):
    pred = np.squeeze(self.pred)
    t = np.squeeze(self.t)
    #test_frames = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    sum_y_pred = np.sum(pred, axis=0).flatten().reshape(t[1].shape)
    sum_y = np.sum(t, axis=0).flatten().reshape(t[1].shape)
    plt.imshow(sum_y_pred-sum_y, cmap='hot', interpolation='nearest')
    plt.show()

  #def animate(self):
  def live_plotter(self,real,pred_,diff,mat1,mat2,mat3,pause_time,identifier=''):
      if mat1==[] and mat2==[] and mat3==[]:
          # this is the call to matplotlib that allows dynamic plotting
          plt.ion()

          fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(13,6))

          # create a variable for the line so we can later update it
          mat1 = ax1.imshow(real)
          mat2 = ax2.imshow(pred_)
          mat3 = ax3.imshow(diff)
          ax1.set_title("iteration")
          ax2.set_title("NN prediction")
          ax3.set_title("difference")
          #update plot label/title
          ax1.grid(False)
          ax2.grid(False)
          ax3.grid(False)

          plt.show()
          plt.show()


      # after the figure, axis, and line are created, we only need to update the y-data
      mat1.set_data(real)
      mat2.set_data(pred_)
      mat3.set_data(diff)
      # adjust limits if new data goes beyond bounds
      # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
      plt.pause(pause_time)
      # return line so we can update it again in the next iteration
      return mat1,mat2,mat3

  def animate(self,pause):
    mat1 = []
    mat2 = []
    mat3 = []
    example = self.spawn()
    prediction = example
    difference = self.spawn()#np.absolute(prediction-example)
    error = 0
    error_frame = np.empty(self.frame_shape)
    while True:

      mat1, mat2, mat3 = self.live_plotter(example,prediction,difference,mat1,mat2,mat3,pause,'Game Of Life')
      for _ in range(self.steps):
        example = evolve(example)
      prediction = self.torus(prediction.reshape(1,prediction.shape[0],prediction.shape[1]))
      #print(prediction.shape)
      prediction = torch.tensor(self.reshape_input(prediction)).float()
      #print(prediction.shape)
      with torch.no_grad():
        prediction = torch.round(self.model(prediction))
      #print(prediction.shape)
      prediction = np.squeeze(np.squeeze((prediction.numpy())))
      #print(prediction.shape)
      difference = np.absolute(prediction-example)
      if np.sum(difference) != 0 and error == 0:
        time.sleep(2)
        error += 1
        error_frame = np.append(error_frame,example)
      if msvcrt.kbhit():
        if ord(msvcrt.getch()) == 27:
          break
    return error_frame
