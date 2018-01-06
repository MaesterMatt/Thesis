#!/usr/bin/env python
__author__ = "Matthew Ng"

import numpy as np
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

cap = cv2.VideoCapture(0)
frame = cv2.imread('hall4.jpg')
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
frame = cv2.GaussianBlur(frame, (3,3),0)
#while(True):
#capture frame by frame
#ret, frame = cap.read()
slopeIntercept = []

#our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
#if lines is None:
#   continue
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
slopeIntercept = [x for x in slopeIntercept if (abs(x[0]) < 3 and abs(x[0]) > 0.1)]

#calculate intercepts
for i, si1 in enumerate(slopeIntercept):
   for si2 in slopeIntercept[i+1:]:
      if abs(si1[0] - si2[0]) > 1:
         cv2.circle(gray, intersect(si1, si2), 10, (0,0,255), -1)


#display the resulting frame
while (True):
   cv2.imshow('frame', gray)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
