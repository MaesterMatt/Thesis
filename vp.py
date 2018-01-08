#!/usr/bin/env python3
__author__ = "Matthew Ng"

import serial
import numpy as np
import time
import cv2
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


#video capture object
cap = cv2.VideoCapture(0)
#width of webcam camera divided by 2
ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
height, width, channels = frame.shape

#serial object
ser = serial.Serial(adp)
time.sleep(3)

while(True):
   #capture frame by frame
   ret, frame = cap.read()
   frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
   frame = cv2.GaussianBlur(frame, (3,3),0)

   slopeIntercept = []
   lineIntersects = []

   #our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(gray, 50, 150)
   lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
   if lines is None:
      print("No Lines!")
      continue
   for i in range(0,min(len(lines), 200)):
      rho,theta = lines[i][0]
   #  for rho, theta in lines[0]:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      if b == 0:
         continue
      slope = -a/b
      intercept = y0 - x0*slope
      #print(slope)
      #print(intercept)
      slopeIntercept.append([slope, intercept])
      #x1 = int(x0 + 1000*(-b))
      #y1 = int(y0 + 1000*(a))
      #x2 = int(x0 - 1000*(-b))
      #y2 = int(y0 - 1000*(a))
      #cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

   #get rid of verticals and horizontals
   slopeIntercept = [x for x in slopeIntercept if (abs(x[0]) < 3 and abs(x[0]) > 0.05)]

   #calculate intercepts
   for i, si1 in enumerate(slopeIntercept):
      for si2 in slopeIntercept[i+1:]:
         if abs(si1[0] - si2[0]) > 0.5:
            lineIntersects.append(intersect(si1, si2))

   numIntersects = len(lineIntersects)
   if numIntersects == 0:
      print("no intersects!")
      continue
   sum_x = sum([x[0] for x in lineIntersects])
   sum_y = sum([y[1] for y in lineIntersects])
   avg_x = int(sum_x/numIntersects)
   avg_y = int(sum_y/numIntersects)
   
   color_x = avg_x if avg_x < width else width
   color_x = color_x if color_x > 0 else 0
   color_x = 2*abs(width/2 - color_x)
   cv2.circle(frame, (avg_x, avg_y), 10, (0, 255-color_x, color_x), -1)
   ser.write(chr(int(color_x)).encode())

   #display the resulting frame
   cv2.imshow('frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break


ser.close()
cap.release()
cv2.destroyAllWindows()
