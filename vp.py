#!/usr/bin/env python3
__author__ = "Matthew Ng"

import serial
import numpy as np
import time
import cv2
import os
from matplotlib import pyplot as plt

def intersect(p1_si, p2_si):
   if len(p1_si) != 2 and len(p2_si) != 2:
      print("Incorrect slope intercept form")
      return
   b = p2_si[1] - p1_si[1]
   s = p1_si[0] - p2_si[0]
   x = b/s
   y = p1_si[0]*x + p1_si[1]
   return (x, y)

#Arduino port #
adp = '/dev/ttyACM0'

#silly stuff
duration = 0.1
freq = 440

#image testing
#frame = cv2.imread('hall1.jpg')

#video capture object
cap = cv2.VideoCapture(0)
#width of webcam camera divided by 2
ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
height, width, channels = frame.shape

#for moving average
oldLines = [[]]
oldIntersects = []

#speed
speed = chr(0)

#serial object
s = False
try:
   ser = serial.Serial(adp)
   time.sleep(3)
   s = True
except:
   print("No Serial Connection")

while(True):
   #capture frame by frame
   ret, frame = cap.read()
   frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
   frame = cv2.GaussianBlur(frame, (3,3),0)
   slopeIntercept = []
   newIntersects = []
   lineIntersects = []

   #our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(gray, 50, 150)
   newLines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)

   if newLines is None:
      if oldLines is None:
         if s:
            ser.write(chr(0).encode())
         continue
      lines = oldLines
      oldLines = None
      print("No Lines!")
   else:
      newLines = newLines.tolist()
      lines = newLines if oldLines is None else newLines + oldLines

   for line in lines:
      for x1,y1,x2,y2 in line:
         cv2.line(frame, (x1,y1), (x2,y2), (0,255,0),2)
         if x2-x1 == 0:
            continue
         slope = (y2-y1)/(x2-x1)
         intercept = y1 - x1*slope
         slopeIntercept.append([slope,intercept])

   #align the horizon line
   slopeIntercept.append([0, height/2])
   cv2.line(frame, (0, round(height/2)), (width, round(height/2)), (0, 0, 255), 2)
   
   slopeIntercept = [x for x in slopeIntercept if (abs(x[0]) < 3 and abs(x[0]) > 0.1) ]
   #[cv2.line(frame, (0, int(x[1])), (1000, int(x[0]*1000 + x[1])), (0,0,255), 2) for x in slopeIntercept]

   #calculate intercepts
   for i, si1 in enumerate(slopeIntercept):
      for si2 in slopeIntercept[i+1:]:
         if abs(si1[0] - si2[0]) > 0.5:
            newIntersects.append(intersect(si1, si2))

   if len(oldIntersects) > 0:
      lineIntersects = lineIntersects + oldIntersects
   if len(newIntersects) > 0:
      lineIntersects = lineIntersects + newIntersects
   
   numIntersects = len(lineIntersects)
   if numIntersects != 0:
      sum_x = sum([x[0] for x in lineIntersects])
      sum_y = sum([y[1] for y in lineIntersects])
      avg_x = int(sum_x/numIntersects)
      avg_y = int(sum_y/numIntersects)

      color_x = avg_x if avg_x < width else width
      color_x = color_x if color_x > 0 else 0

      #print(int(255/width*color_x))
      speed = chr(int(255/width*color_x))

      color_x = 2*abs(width/2 - color_x)
      #this is where the avg circle is drawn
      cv2.circle(frame, (avg_x, avg_y), 10, (0, 255-color_x, color_x), -1)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   #display the resulting frame
   cv2.imshow('frame', frame)
   #Send location of VP to Arduino
   if s:
      ser.write(speed.encode())

   #adjust the moving average
   if newLines is not None:
      oldLines = newLines
   oldIntersects = newIntersects

if s:
   ser.write(chr(0).encode())
   ser.close()
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
time.sleep(0.25)
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq*2))
cap.release()
cv2.destroyAllWindows()
