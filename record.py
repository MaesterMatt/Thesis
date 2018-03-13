#!/usr/bin/env python3
__author__ = "Matthew Ng"

import serial
import numpy as np
import time
import cv2
import os
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import imutils
import csv

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
counter = 0

while(cap.isOpened()):
   ret, frame = cap.read()
   
   out.write(frame)

   cv2.imshow('frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   if counter >= 2000:
      break
   counter += 1

cap.release()
out.release()
