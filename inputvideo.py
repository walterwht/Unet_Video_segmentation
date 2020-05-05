"""
A Simple PyTorch Video Dataset Class for loading videos using PyTorch
Dataloader. This Dataset assumes that video files are Preprocessed
 by being trimmed over time and resizing the frames.


If you find this code useful, please star the repository.
"""

from __future__ import print_function, division

import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_video(video_file, channel, times,videosize):
        # Open the video file
  cap = cv2.VideoCapture(video_file)
  frames = torch.FloatTensor( times, channel, videosize, videosize)

  for f in range(times):

    ret, frame = cap.read()
    if ret:
      frame = torch.from_numpy(frame)
      # HWC2CHW  
      frame = frame.permute(2, 0, 1)
      frames[f, :, :, :] = frame*-1

    else:
      print("Skipped!")
      break

  return frames


