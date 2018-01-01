#!/usr/bin/env python
__author__ = "Matthew Ng"

import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

print("hello")
while(True):
   #capture frame by frame
   ret, frame = cap.read()

   #our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   edges = cv2.Canny(gray, 100, 200)
   lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
   print(len(lines))
   for rho, theta in lines[0]:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))
      cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

   #display the resulting frame
   cv2.imshow('frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()


