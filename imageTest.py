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

def intersect(p1_si, p2_si):
   if len(p1_si) != 2 and len(p2_si) != 2:
      print("Incorrect slope intercept form")
      return
   b = p2_si[1] - p1_si[1]
   s = p1_si[0] - p2_si[0]
   x = b/s
   y = p1_si[0]*x + p1_si[1]
   return (x, y)

def writeListToCSV(data):
   with open("eric.csv", 'w') as myfile:
      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
      wr.writerow([int(x) for x in data])
   
csvlist = []

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (320, 240))
#hog oink
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while (cap.isOpened()):
   #image testing
   #frame = cv2.imread('hall4.jpg')
   ret, frame = cap.read()

   (rects, weights) = hog.detectMultiScale(frame, winStride=(8,8),padding=(16,16),scale=1.05)
   rects = np.array([[x/2,y/2,(x+w)/2,(y+h)/2] for (x,y,w,h) in rects])
   pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
   pick = []
   frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
   height, width, channels = frame.shape
   frame = cv2.GaussianBlur(frame, (3,3),0)
   slopeIntercept = []
   newIntersects = []
   lineIntersects = []

   #our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   edges = cv2.Canny(gray, 50, 150)
   for(x1,y1,x2,y2) in pick:
      cv2.rectangle(edges, (x1,y1), (x2,y2), 0,thickness=-1)
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=(50 - len(pick)*3), minLineLength=50, maxLineGap=30)

   for xa,ya,xb,yb in pick:
      lines = [[(x1,y1,x2,y2) for x1,y1,x2,y2 in line if not ((xa < x1 and x1 < xb and xa < x2 and x2 < xb)and(ya<y1 and y1<yb and ya<y2 and y2<yb)) ] for line in lines]

   for line in lines:
      for x1,y1,x2,y2 in line:
         cv2.line(frame, (x1,y1), (x2,y2), (0,255,0),2)
         if x2-x1 == 0:
            continue
         slope = (y2-y1)/(x2-x1)
         intercept = y1 - x1*slope
         slopeIntercept.append([slope,intercept])

   #align the horizon line
   #slopeIntercept.append([0, height/2])
   cv2.line(frame, (0, round(height/2)), (width, round(height/2)), (0, 0, 255), 2)

   slopeIntercept = [x for x in slopeIntercept if (abs(x[0]) > 0.1 and abs(x[0]) < 9)] #eliminate lines close to zero
   [cv2.line(frame, (0, int(x[1])), (1000, int(x[0]*1000 + x[1])), (0,0,255), 2) for x in slopeIntercept]

   #calculate intersects
   for i, si1 in enumerate(slopeIntercept):
      for si2 in slopeIntercept[i+1:]:
         if abs(si1[0] - si2[0]) > 0.5:
            lineIntersects.append(intersect(si1, si2))

   #gets rid of intersects in the top 1/3 or bottom 1/3 of the image
   lineIntersects = [li for li in lineIntersects if (li[1] > height/3 and li[1] < height*2/3)]

   [cv2.circle(frame, (int(intersect[0]), int(intersect[1])), 10, (255, 0, 0), -1) for intersect in lineIntersects]

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
      cv2.circle(frame, (avg_x, avg_y), 10, (0, 255, 0), -1)
      csvlist.append(avg_x)

   for(x1,y1,x2,y2) in pick:
      cv2.rectangle(frame, (x1,y1), (x2,y2), (25, 25, 25),thickness=2)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   #display the resulting frame
   cv2.imshow('frame', frame)
   out.write(frame)


writeListToCSV(csvlist)
cap.release()
out.release()
cv2.destroyAllWindows()
