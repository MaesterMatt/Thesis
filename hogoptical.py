#!/usr/bin/env python3
'''Part of this code was taken from alduxvm on github
people-detection.py - rpi-opencv/people-detection.py 
'''
HOG = False

import time
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils

hog = cv2.HOGDescriptor()
cap = cv2.VideoCapture('yavishtwalk.avi')#'stationary.avi')
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

clamp = lambda x, minn, maxx: max(min(maxx, x), minn)

#farneflow
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (320, 240))
height, width = frame1.shape[:2]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
ROI = []
numdetect = 0
avgWin = 1
while (cap.isOpened()):
   start_time = time.time()
   #frame = cv2.imread('../Thesis/hall4.jpg')
   ret, image = cap.read()
   image = cv2.resize(image, (320, 240))
   nxt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 10, 20, 3, 5, 1.2, 2)
   x_flo = flow[...,0]
   y_flo = flow[...,1]
   orig = image.copy()

   (rect, weight) = hog.detectMultiScale(image, winStride=(8,8),padding=(16,16), scale=1.12)
   for i in range(0,len(ROI)):
      try:
         rect = rect.tolist()
      except:
         pass
      try:
         weight = weight.tolist()
      except:
         pass
      subim = image[ROI[i][1]:ROI[i][3], ROI[i][0]:ROI[i][2]]
      if HOG:
         r = []
         w = []
      else:
         r,w = hog.detectMultiScale(image[ROI[i][1]:ROI[i][3], ROI[i][0]:ROI[i][2]] , winStride=(4,4), padding=(8,8), scale=1.12)
         cv2.rectangle(orig, (ROI[i][0], ROI[i][1]), (ROI[i][2], ROI[i][3]), (255, 25, 0), 1)
         

      if(len(r) > 0):
         if len(rect) > 0:
            [rect.append([x+ROI[i][0], y+ROI[i][1], w, h]) for (x,y,w,h) in r]
            #print('b4-{}_{}'.format(r, w[0][0]))
            w = [x*2 for x in w]
            [weight.append(x.tolist()) for x in w]
            print('xx-{}_{}'.format(rect, weight))
         else:
            rect = [[x+ROI[i][0], y+ROI[i][1], w, h] for (x,y,w,h) in r]
            weight = w*2
      else:
         print("Sadfaces")
   ROI = []

   #print('{}_{}'.format(rect, weight))
   rects = []
   for i, (x,y,w,h) in enumerate(rect):
      if weight[i][0] > 1:
         rects.append(rect[i])

   rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
   pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

   # Draw the Rectangle on the Image
   for (xA, yA, xB, yB) in pick:
      
      cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 25, 0), 2)
      numdetect += 1
      xSubFlo = x_flo[yA:yB, xA:xB]
      ySubFlo = y_flo[yA:yB, xA:xB]
      avgX = sum(sum(xSubFlo))/abs((yB-yA)*(xB-xA))
      avgY = sum(sum(ySubFlo))/abs((yB-yA)*(xB-xA))

      '''
      y = (yA+yB)//2
      x = (xA+xB)//2
      xSubFlo = x_flo[y-avgWin:y+avgWin, x-avgWin:x+avgWin]
      ySubFlo = y_flo[y-avgWin:y+avgWin, x-avgWin:x+avgWin]
      avgX = sum(sum(xSubFlo))//pow((2*avgWin+1),2) - avgX
      avgY = sum(sum(ySubFlo))//pow((2*avgWin+1),2) - avgY
      '''
      #ROI.append((xA, yA, xB, yB))

      xA = int(clamp(min(xA+avgX,xA), 0, width))
      xB = int(clamp(max(xB+avgX,xB), 0, width))
      yA = int(clamp(min(yA+avgY,yA), 0, height))
      yB = int(clamp(max(yB+avgY,yB), 0, height))

      ROI.append((xA, yA, xB, yB))
      
      p1 = ((xB+xA)//2, (yB+yA)//2) 
      p2 = (int(p1[0] + avgX*5), p1[1])
      cv2.arrowedLine(orig, p1, p2, (255,255,0), 1, 4, 0, 0.1)
      #print('left' if p2[0] < p1[0] else 'right')
      
      #print('{}_{}'.format(p1, p2))

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   orig = cv2.resize(orig, (0,0), fx=3, fy=3)
   cv2.imshow('frame', orig)
   print(1/(time.time()-start_time))
   #time.sleep(0.125)
cap.release()
print(numdetect)
cv2.destroyAllWindows()
