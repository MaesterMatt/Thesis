#!/usr/bin/env python3
'''Large Portion of this code was taken from alduxvm on github
people-detection.py - rpi-opencv/people-detection.py 
'''
import time
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils

hog = cv2.HOGDescriptor()
cap = cv2.VideoCapture('../Thesis/yavishtwalk.avi')
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

clamp = lambda x, minn, maxx: max(min(maxx, x), minn)

#farneflow
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (320, 240))
height, width = frame1.shape[:2]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
ROI = []
numdetect = 0
while (cap.isOpened()):
   start_time = time.time()
   #frame = cv2.imread('../Thesis/hall4.jpg')
   ret, frame = cap.read()
   image = cv2.resize(frame, (320, 240))
   nxt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 1, 20, 3, 5, 1.2, 2)
   x_flo = flow[...,0]
   y_flo = flow[...,1]
   orig = image.copy()

   (rect, weight) = hog.detectMultiScale(image, winStride=(8,8),padding=(16,16), scale=1.08)
   for i in range(0,len(ROI)):
      try:
         rect = rect.tolist()
         weight = weight.tolist()
      except:
         pass
      subim = image[ROI[i][1]:ROI[i][3], ROI[i][0]:ROI[i][2]]
      r,w = hog.detectMultiScale(image[ROI[i][1]:ROI[i][3], ROI[i][0]:ROI[i][2]] , winStride=(4,4), padding=(8,8), scale=1.05)
      r = []

      if(len(r) > 0):
         if len(rect) > 0:
            [rect.append([x+ROI[i][0], y+ROI[i][1], w, h]) for (x,y,w,h) in r]
            [weight.append(x) for x in w.tolist()]
         else:
            print("beluga")
            rect = [[x+ROI[i][0], y+ROI[i][1], w, h] for (x,y,w,h) in r]
            weight = w
         #try:
         #weight.append(w)
         #except:
         #   print("NOOP")
         #   rects = r
         #   weights = w
   ROI = []

   print('{}_{}'.format(rect, weight))
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

      xA = int(clamp(xA+avgX, 0, width))
      xB = int(clamp(xB+avgX, 0, width))
      yA = int(clamp(yA+avgY, 0, height))
      yB = int(clamp(yB+avgY, 0, height))
      
      ROI.append((xA, yA, xB, yB))
      #p1 = ((xB+xA)//2, (yB+yA)//2) 
      #p2 = (int(p1[0] + avgX), int(p1[1] + avgY))
      #print('{}_{}'.format(p1, p2))

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break
   image = cv2.resize(orig, (0,0), fx=3, fy=3)
   cv2.imshow('frame', image)
   print(numdetect)
cap.release()
cv2.destroyAllWindows()
