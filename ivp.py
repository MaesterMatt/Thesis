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
import math
import statistics
from skimage import measure

HOG = False
HOGOPTICAL = False
KM=True
RECORD = False
filename = 'empty_rotate.'
height = 240
width = 320

def intersectWeight(mean, var, val):
   return (1/(math.sqrt(2*math.pi*var)))*math.pow(math.e, -math.pow(val-mean, 2)/(2*var))

def intersect(p1_si, p2_si):
   if len(p1_si) != 2 and len(p2_si) != 2:
      print("Incorrect slope intercept form")
      return
   b = p2_si[1] - p1_si[1]
   s = p1_si[0] - p2_si[0]
   x = b/s
   y = p1_si[0]*x + p1_si[1]
   return (x, y)

def writeListToCSV(data, filename):
   with open(filename, 'w') as myfile:
      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
      wr.writerow([int(x) for x in data])

def bottomHalfImage(image):
   global half_height
   cropped_im = image[half_height:height]
   return cropped_im

def topHalfImage(image):
   global half_height
   cropped_im = image[0:half_height]
   return cropped_im

kmeanC = width//2
def floorSeg(fin):
   global kmeanC
   fin = cv2.medianBlur(fin, 21)
   bh=bottomHalfImage(fin)
   Z = bh.reshape((bh.shape[0]*bh.shape[1]))

   Z = np.float32(Z)

   criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
   K = 5
   ret, label, center = cv2.kmeans(Z,K, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)


   tempLabel = label.reshape(height//2, width, 1)
   #print(tempLabel[height//2-1, width//2])
   B = Z[label.ravel()==1]
   center = np.uint8(center)
   res = center[label.flatten()]
   res = np.zeros((len(res), 1))
   res[label.ravel()==tempLabel[height//2-1, kmeanC]] = 1
   res2 = res.reshape(bh.shape)
   thresh = cv2.threshold(res2, 0.5,255,cv2.THRESH_BINARY)[1]

   labels = measure.label(thresh, neighbors=8, background=0)
   mask = np.zeros(thresh.shape,dtype="uint8")

   largestMask = None
   maxNum = -1
   for lab in np.unique(labels):
      if lab==0:
         continue
      labelMask = np.zeros(thresh.shape, dtype="uint8")
      labelMask[labels == lab] = 255
      numPixels = cv2.countNonZero(labelMask)
      if numPixels > maxNum:
         maxNum = numPixels
         largestMask = labelMask
   if largestMask is not None:
      cnts = cv2.findContours(largestMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if imutils.is_cv2() else cnts[1]

      peak = (kmeanC, height)
      for i in cnts[0]:
         if i[0][1] < peak[1]:
            peak = (i[0][0], i[0][1])
      kmeanC = peak[0]
      
      #cv2.imshow('res2',largestMask)

left = (-0.5, 0)
right = (0.5, width)
def floorCalc(frame):
   global slopeIntercept, inter, left, right

   #intersection Threshold
   interThresh = 4

   half = bottomHalfImage(frame)
   half_gray = cv2.cvtColor(half, cv2.COLOR_RGB2GRAY)
   half_edge = cv2.Canny(half_gray, 50, 150)
   half_line = cv2.HoughLinesP(half_edge, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)

   if half_line is None:
      return
   for line in half_line:
      for x1, y1, x2, y2 in line:
         if x2-x1 == 0 or y2-y1 == 0:
            continue
         slope = (y2-y1)/(x2-x1)
         intercept = y1-x1*slope
         slopeIntercept.append([slope,intercept])


   #[print(abs((-x[1])/x[0]-MovingAverage)) for x in slopeIntercept]
   slopeIntercept = [x for x in slopeIntercept if (abs(-x[1]/x[0]-MovingAverage)  < 100)] #eliminate lines that don't meet the VP
   slopeIntercept = sorted(slopeIntercept)
   siPos = [x for x in slopeIntercept if x[0] > 0]
   siNeg = [x for x in slopeIntercept if x[0] < 0]

   flatLines = [x for x in slopeIntercept if abs(x[0]) < 0.3]
   if len(flatLines) > interThresh:
      if inter == True:
         print("INTERSECTION? - Lots of Flats")
      inter = True
   else:
      inter = False

   # Gets the two most likely ground lines and uses them to find position
   slopeIntercept = []
   rBoo = False
   if len(siPos) > 0:
      x = min(range(len(siPos)), key=lambda i: abs(siPos[i][0] - right[0]))
      if abs(siPos[x][0]-right[0]) < 0.3:
         slopeIntercept.append([siPos[x][0],siPos[x][1] + half_height])
         right = (siPos[x][0],siPos[x][1] + half_height)
         rBoo = True
      else:
         slopeIntercept.append([right[0],right[1]])
   else:
      slopeIntercept.append([right[0],right[1]])

   lBoo = False
   if len(siNeg) > 0:
      y = min(range(len(siNeg)), key=lambda i: abs(siNeg[i][0] - left[0]))
      if abs(siNeg[y][0]-left[0]) < 0.3:
         slopeIntercept.append([siNeg[y][0],siNeg[y][1]+half_height])
         left = (siNeg[y][0],siNeg[y][1]+half_height)
         lBoo = True
      else:
         slopeIntercept.append([left[0],left[1]])
   else:
      slopeIntercept.append([left[0],left[1]])

   # calculate floor based on ground lines
   [cv2.line(frame, (0, int(x[1])), (1000, int(x[0]*1000 + x[1])), (255,0,255), 2) for x in slopeIntercept]
   
   if not lBoo and not rBoo:
      left = (-0.5, 0)
      right = (0.5, width)
   return slopeIntercept

clamp = lambda x, minn, maxx: max(min(maxx, x), minn)

##############################################
################## MAIN ######################
##############################################
cap = cv2.VideoCapture(filename + 'avi')#'yavishtwalk.avi')#'empty_rotate.avi')

csvlist = []
timelist = []
MA = [160]*5
MovingAverage = 160

# For driving adjustments
numwin = 7
imBin = [0]*numwin # for tracking stability
lAdj = 0
rAdj = 0

# Arduino Drive Port Setup#
try: 
   adp = '/dev/ttyACM0'
   ser = serial.Serial(adp)
   ser.setBaudrate(115200)
   time.sleep(3)
   s = True
except:
   s = False
   print("No Serial Connection")

if RECORD:
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

ret, frame1 = cap.read()
if HOG or HOGOPTICAL:
   hog = cv2.HOGDescriptor()
   hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
   if HOGOPTICAL:
      frame1 = cv2.resize(frame1,(width,height))
      prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
      ROI = []
#to end the stream after # of frames
counter = 2000

#For intersection detection
inter = False

start_time = time.time()
###########################################################
#######################LOOP################################
###########################################################
while (cap.isOpened()):
   framelock_start = time.time()
   ret, frame = cap.read()
   if frame is None:
      break
   frame = cv2.resize(frame, (width, height))

   if HOG or HOGOPTICAL:
      (rect, weight) = hog.detectMultiScale(frame, winStride=(8,8),padding=(16,16),scale=1.05)
      try:
         rect = rect.tolist()
      except:
         pass
      try:
         weight = weight.tolist()
      except:
         pass
      if HOGOPTICAL:
         nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 10, 20, 3, 5, 1.2, 2) 
         x_flo = flow[...,0]
         y_flo = flow[...,1]
         for i in range(0, len(ROI)):
            r,w = hog.detectMultiScale(frame[ROI[i][1]:ROI[i][3], ROI[i][0]:ROI[i][2]], winStride=(8,8), padding=(16,16), scale=1.05)
            
            if(len(r) > 0):
               if len(rect) > 0:
                  [rect.append([x+ROI[i][0], y+ROI[i][1], w, h]) for (x,y,w,h) in r]
                  w = [x*2 for x in w]
                  [weight.append(x.tolist()) for x in w]
               else:
                  rect = [[x+ROI[i][0], y+ROI[i][1], w, h] for (x,y,w,h) in r]
                  weight = [x*2 for x in w]
         ROI = []
      rects = []
      for i, (x,y,w,h) in enumerate(rect):
         if weight[i][0] > 1:
            rects.append(rect[i])

      rects = np.array([[x,y,(x+w),(y+h)] for (x,y,w,h) in rects])
      pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
   slopeIntercept = []
   newIntersects = []
   lineIntersects = []
   
   #Half height calculation
   half_height = round(height/2)

   slopeIntercept = floorCalc(frame)
   if slopeIntercept is None:
      slopeIntercept = []
   
   #our operations on the frame come here
   gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   blurgray = cv2.medianBlur(gray, 3)
   high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   low_thresh = 0.5*high_thresh
   edges = cv2.Canny(gray, low_thresh, high_thresh)
   lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
   if lines is None:
      continue
   templin = []
   [[templin.append([x1,y1,x2,y2]) for x1,y1,x2,y2 in line] for line in lines]
   lines = templin

   if HOG or HOGOPTICAL:
      for xa,ya,xb,yb in pick:
         newLin = []
         for x1,y1,x2,y2 in lines:
            if not ((xa < x1 and x1 < xb) and (ya < y1 and y1 < yb)) or not ((xa < x2 and x2 < xb) and (ya < y2 and y2 < yb)):
               newLin.append([x1,y1,x2,y2])

         lines = newLin
         if HOGOPTICAL:
            xSubFlo = x_flo[ya:yb,xa:xb]
            ySubFlo = y_flo[ya:yb,xa:xb]
            avgX = sum(sum(xSubFlo))/abs((yb-ya)*(xb-xa))
            avgY = sum(sum(ySubFlo))/abs((yb-ya)*(xb-xa))

            xA = int(clamp(min(xa+avgX,xa), 0, width))
            xB = int(clamp(max(xb+avgX,xb), 0, width))
            yA = int(clamp(min(ya+avgY,ya), 0, height))
            yB = int(clamp(max(yb+avgY,yb), 0, height))

            ROI.append((xA, yA, xB, yB))
   for x1,y1,x2,y2 in lines:
      if x2-x1 == 0:
         continue
      slope = (y2-y1)/(x2-x1)
      intercept = y1 - x1*slope
      slopeIntercept.append([slope,intercept])

   #Place the horizon line to assist with stability
   slopeIntercept.append([0, height/2])
   cv2.line(frame, (0, round(height/2)), (width, round(height/2)), (0, 0, 255), 2)

   slopeIntercept = [x for x in slopeIntercept if (abs(x[0]) > 0.1 and abs(x[0]) < 9)] #eliminate lines close to zero and too strong
   #[cv2.line(frame, (0, int(x[1])), (1000, int(x[0]*1000 + x[1])), (0,0,255), 2) for x in slopeIntercept]

   #calculate intersects
   for i, si1 in enumerate(slopeIntercept):
      for si2 in slopeIntercept[i+1:]:
         if abs(si1[0] - si2[0]) > 0.5:
            lineIntersects.append(intersect(si1, si2))

   #gets rid of intersects in the top 1/3 or bottom 1/3 of the image
   #lineIntersects = [li for li in lineIntersects if (li[1] > height/3 and li[1] < height*2/3)]

   [cv2.circle(frame, (int(intersect[0]), int(intersect[1])), 10, (255, 0, 0), -1) for intersect in lineIntersects]

   color = (0,255,0)
   if KM:
      floorSeg(blurgray)
      avg_x = kmeanC
   
   if not KM or abs(kmeanC-MovingAverage) > 30:
      color = (0, 255, 255)
      numIntersects = len(lineIntersects)
      
      if numIntersects != 0:
         sum_x = sum([(x[0]-MovingAverage)*intersectWeight(MovingAverage, width, x[0]) for x in lineIntersects])
         
         #recalculate centerpoint because gaussian is off!
         if abs(sum_x) < 1:
            color = (255,0,255)
            sum_x = sum([x[0] for x in lineIntersects])
            avg_x = int(sum_x//numIntersects)
         else:
            avg_x = int(MovingAverage + sum_x)
      else:
         color = (0, 0, 255)
         avg_x = int(MovingAverage)


   #update moving average
   del MA[0]
   MA.append(avg_x)      
   MovingAverage = sum(MA)/len(MA)
   csvlist.append(MovingAverage)
   kmeanC = avg_x
   #this is where the avg circle is drawn
   cv2.circle(frame, (int(MovingAverage), round(height/2)), 10, color, -1)

   if s:         
      winsize = width/numwin
      left = 8 + lAdj
      right = 8 + rAdj
      bothAdj = 0

      thisBin = -1
      for i in range(1,numwin+1):
         if MovingAverage < i * winsize:
            thisBin = i-1
            break

      if thisBin == -1:
         print("WOAH NEGATIVE ERROR!!!")
      else:
         if sum(imBin) > 30:
            diff = sum(imBin[0:numwin//2]) - sum(imBin[numwin//2+1:numwin])
            if abs(diff) > 2:
               lAdj -= diff/abs(diff)
               rAdj += diff/abs(diff)
               lAdj = clamp(lAdj, -1, 1)
               rAdj = clamp(rAdj, -1, 1)
            else:
               lAdj = 0
               rAdj = 0
            imBin = [0]*numwin
            
         imBin[thisBin] += 1            
         bothAdj = thisBin - numwin//2
         left += bothAdj
         right -= bothAdj

         left = int(clamp(left, 0, 15))
         right = int(clamp(right, 0, 15))
            
         print("left {}, right {}".format(left, right))
         ser.write(chr((left << 4 | right)).encode())

   if HOG or HOGOPTICAL:
      for(x1,y1,x2,y2) in pick:
         cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 0),thickness=2)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

   if counter <= 0:
      break

   #display the resulting frame
   cv2.imshow('frame', frame)
   counter -= 1
   if RECORD:
      out.write(frame)

   #timelist.append(time.time() - start_time)
   #delta_time = time.time() - framelock_start
   #time.sleep(abs(0.0416-delta_time))

if s:
   ser.write(chr(0).encode())
   ser.close()

if RECORD:
   out.release()

print('{} FPS'.format(int((2000-counter)/(time.time() - start_time))))
with open(filename + 'csv', 'r') as f:
   reader = csv.reader(f)
   lines = list(reader)

diff = [abs(float(x)/2 - float(y)) for x,y in zip(lines[0], csvlist)]
print(sum(diff)/len(diff))
print(statistics.variance(csvlist))
print(statistics.variance([float(x) for x in lines[0]]))
cap.release()
cv2.destroyAllWindows()
