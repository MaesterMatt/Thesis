#!/usr/bin/env python3
__author__ = "Matthew Ng"

import argparse
import serial
import numpy as np
import time
import cv2
import os
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import imutils
import csv


storePoint = []
freeze = False
def click_and_save(event, x, y, flags, param):
   global storePoint, freeze
   if event == cv2.EVENT_LBUTTONDOWN:
      print(x)
      freeze = False
      storePoint.append(int(x))
         
def writeListToCSV(data, filename):
   with open(filename, 'w') as myfile:
      wr = csv.writer(myfile)
      wr.writerow([int(x) for x in data])

cap = cv2.VideoCapture('chairdrive2.avi')
counter = 0
width = 640
height = 480
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_and_save)

while(cap.isOpened()):
   if not freeze:
      ret, frame = cap.read()
   freeze = True
   if frame is None:
      break

   cv2.line(frame, (0, round(height/2)), (width, round(height/2)), (0, 0, 255), 2)
   cv2.imshow('frame', frame)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

print(storePoint)
writeListToCSV(storePoint, "eric.csv")
cap.release()
