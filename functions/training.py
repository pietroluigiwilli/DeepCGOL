### Training functions:  The functions contained within this document are used to trian the neural network and to assess its performance.
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import torch
import torchvision
from tqdm import tqdm


def accuracy(y, x, t,steps=1):
  pred_acc = 1-(torch.sum(torch.abs(y-t))/torch.numel(y))
  data_acc = 1-(torch.sum(torch.abs(x-torch.nn.functional.pad(t,(steps,steps,steps,steps))))/torch.numel(t))
  return data_acc, pred_acc

def train_simple(network, x, y, epochs, suffix, steps,lr=0.01, dev="cpu", rounding=False):
  X = torch.tensor(x)
  Y = torch.tensor(y)
  device = torch.device(dev)
  network1 = network.to(device)
  loss = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(params=network1.parameters(), lr=lr)
  learning_plot = []
  accuracy_plot = []


  train_index = int(len(X)*(25/30))
  val_index = int(len(X)*(275/300))
  val_load_len = val_index-train_index
  B_size = 50


  for epoch in tqdm(range(epochs)):
    for i in tqdm(range(train_index//B_size)):
      x = X[i:i+B_size].float()
      t = Y[i:i+B_size].float()
      optimizer.zero_grad()
      y = network1(x.to(device))
      if rounding == True:
        y = torch.round(y)
      J = loss(y.to(device), t.to(device))
      learning_plot.append(J.item())
      J.backward()
      optimizer.step()

    with torch.no_grad():
      j_val = 0
      d_acc = 0
      p_acc = 0
      for i in tqdm(range((train_index//B_size),(val_index//B_size))):
        x = X[i:i+B_size].float()
        t = Y[i:i+B_size].float()
        y = network1(x.to(device))
        data_acc, pred_acc = accuracy(y.to(device),x.to(device),t.to(device), steps)
        accuracy_plot.append(pred_acc.item())
        j = loss(y.to(device), t.to(device))
        learning_plot.append(j.item())
        d_acc += data_acc.item()
        p_acc += pred_acc.item()
        j_val += j.item()
    print(f"epoch {epoch}, Loss = {j_val/(val_load_len/B_size)}")
    print(f"inherrent similarity = {d_acc*100/(val_load_len/B_size):1.4f} %, prediction accuracy = {p_acc*100/(val_load_len/B_size):1.4f} %")
    torch.save(network1.state_dict(), f"C:/Users/Pietro Willi/Desktop/Python Project/MACHINE LEARNING/Jewel2/model_weights/GOL_params_{suffix}_{steps}step.pth")
  fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
  ax1.plot(learning_plot, label="loss")
  #plt.legend()
  ax2.plot(accuracy_plot,label="accuracy")
  #plt.legend()
  return network1
