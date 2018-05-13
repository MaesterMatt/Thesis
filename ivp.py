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
KM= True
RECORD = False
GAUS = True
FLOOR = True
filename = 'RealWorldTesting/cs.'
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

def km1d(mylist):
   centroid = 0
   newcentroid = len(mylist)//2
   stopper = []
   mylist = sorted(mylist)
   while abs(centroid - newcentroid) > 1:
      centroid = newcentroid
      sum_x = sum([(x-mylist[centroid])*intersectWeight(mylist[centroid], max(mylist)-min(mylist), x) for x in mylist])
      newcentroid += int(sum_x/abs(sum_x))
   return mylist[newcentroid]
      

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
   global kmeanC, left, right
   fin = cv2.medianBlur(fin, 21)
   gray = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
   fin = cv2.medianBlur(gray, 21)
   bh=bottomHalfImage(fin)
   Z = bh.reshape((bh.shape[0]*bh.shape[1]))

   Z = np.float32(Z)

   criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
   K = 7
   ret, label, center = cv2.kmeans(Z,K, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)


   tempLabel = label.reshape(height//2, width, 1)
   B = Z[label.ravel()==1]
   center = np.uint8(center)
   res = center[label.flatten()]
   res = np.ones((len(res), 1))
   res[label.ravel()==tempLabel[height//2-1, kmeanC]] = 0#1
   res2 = res.reshape(bh.shape)
   thresh = cv2.threshold(res2, 0.5,255,cv2.THRESH_BINARY)[1]

   labels = measure.label(thresh, neighbors=8, background=1)#0
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
      cnts = cv2.findContours(largestMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if imutils.is_cv2() else cnts[1]

      peak = (kmeanC, height)

      for i in cnts[0]:
         if i[0][1] < peak[1] and i[0][1]:
            peak = (i[0][0], i[0][1])
      kmeanC = peak[0]
      #M = np.float32([[1,0,0],[0,1, height//2]])
      #dst = cv2.warpAffine(largestMask, M, (width, height))

      return largestMask
      #cv2.imshow('res2',largestMask)

left = (-0.5, 0)
right = (0.5, width)
def floorCalc(frame):
   global slopeIntercept, inter, left, right, nearWallAdj

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

   slopeIntercept = [x for x in slopeIntercept if (abs(-x[1]/x[0]-MovingAverage)  < 100)] #eliminate lines that don't meet the VP
   slopeIntercept = sorted(slopeIntercept)
   siPos = [x for x in slopeIntercept if x[0] > 0]
   siNeg = [x for x in slopeIntercept if x[0] < 0]

   #flatLines = [x for x in slopeIntercept if abs(x[0]) < 0.3]
   #if len(flatLines) > interThresh:
   #   if inter == True:
   #      print("INTERSECTION? - Lots of Flats")
   #   inter = True
   #else:
   #   inter = False

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

   # draw floor based on ground lines
   if len(slopeIntercept) == 2:
      i = intersect(slopeIntercept[0],slopeIntercept[1])
      cv2.line(frame, (0, int(left[1])), (int(i[0]), int(i[1])), (255,255,255),2)
      cv2.line(frame, (width, int(width*right[0] + right[1])), (int(i[0]), int(i[1])),(255,255,255),2)
   l = abs(math.atan(left[0])*180/math.pi)
   r = math.atan(right[0])*180/math.pi

   if lBoo and rBoo:
      if abs(l-r) > 20: 
         if l < r:
            nearWallAdj = -2
            print('too close to right! left = {}'.format(l))
         else:
            nearWallAdj = 2
            print('too close to left! right = {}'.format(r))
      else:
         #print(abs(l-r))
         nearWallAdj = 0
   
   
   if not lBoo and not rBoo:
      left = (-0.5, 0)
      right = (0.5, width)
   return slopeIntercept

def turnDrive(direction, severity, rt):
   global s, ret, frame
   if s:
      #rotate
      print("ROtate")
      for i in range (0, severity//20):
         if direction > 0:
            rMotor = 8
            lMotor = 0
         else:
            rMotor = 0
            lMotor = 8
         ser.write(chr((lMotor << 4 | rMotor)).encode())
         time.sleep(rt)
      #drive straight
      for i in range(0, severity//10):
         rMotor = 8
         lMotor = 8
         ser.write(chr((lMotor << 4 | rMotor)).encode())
         time.sleep(rt)
      #reorient
      print("ROtate back")
      for i in range (0, severity//20):
         if direction > 0:
            rMotor = 0
            lMotor = 8
         else:
            rMotor = 8
            lMotor = 8
         ser.write(chr((lMotor << 4 | rMotor)).encode())
         time.sleep(rt)
      ret, frame = cap.read()
      frame = cv2.resize(frame, (width, height))

clamp = lambda x, minn, maxx: max(min(maxx, x), minn)

##############################################
################## MAIN ######################
##############################################
#cap = cv2.VideoCapture(filename + 'avi')
cap = cv2.VideoCapture(0)#'yavishtwalk.avi')#'empty_rotate.avi')

csvlist = []
timelist = []
MA = [160]*5
MovingAverage = 160

# For driving adjustments
numwin = 5
imBin = [0]*numwin # for tracking stability
Adj = 0
nearWallAdj = 0
personAvoidance = 0
personAvoidCount = 0

# Arduino Drive Port Setup#
try: 
   adp = '/dev/ttyACM0'
   ser = serial.Serial(adp, baudrate=115200)
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
count = 0

#For intersection detection
inter = False

start_time = time.time()
###########################################################
#######################LOOP################################
###########################################################
runTime = 0
while (cap.isOpened()):
   framelock_start = time.time()
   ret, frame = cap.read()
   if frame is None:
      break
   frame = cv2.resize(frame, (width, height))
   img = frame.copy()

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

   if FLOOR:
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

   runTime = time.time() - framelock_start
   if HOG or HOGOPTICAL:
      if len(pick) == 0 and personAvoidCount == 0:
         personAvoidance = 0
      for xa,ya,xb,yb in pick:
         personLoc = int((xa+xb)/2 - MovingAverage)
         if not personLoc:
            personLoc += 1
         personAvoidance = personLoc//abs(personLoc)
         mult = 1 + (1 if yb > 200 else 0)
         personAvoidance = personAvoidance * mult * 2
         if personAvoidCount == 0:
            personAvoidCount = 20
            turnDrive(personAvoidance, yb, runTime)
         newLin = []
         for x1,y1,x2,y2 in lines:
            #if not ((xa < x1 and x1 < xb) and (ya < y1 and y1 < yb)) and not ((xa < x2 and x2 < xb) and (ya < y2 and y2 < yb)):
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
      #print("Person Avoidance = {}".format(personAvoidance))
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
            inter = intersect(si1,si2)
            if inter[0] > 0 and inter[0] < width and inter[1] > 0 and inter[1] < height:
               lineIntersects.append(inter)

   #gets rid of intersects in the top 1/3 or bottom 1/3 of the image
   #lineIntersects = [li for li in lineIntersects if (li[1] > height/3 and li[1] < height*2/3)]

   [cv2.circle(frame, (int(intersect[0]), int(intersect[1])), 10, (255, 0, 0), -1) for intersect in lineIntersects]

   color = (0,255,0)
   if KM:
      mask = floorSeg(img)
      bh = bottomHalfImage(frame)
      bh = cv2.bitwise_and(bh, bh, mask = mask)
      frame[height//2:height] = bh

   sum_x = 0
   numIntersects = len(lineIntersects)
   if numIntersects != 0:
      if GAUS:
         avg_x = int(km1d([x[0] for x in lineIntersects]))
      else:
         avg_x = int(sum([x[0] for x in lineIntersects])/numIntersects)
   else:
      color = (0, 0, 255)
      avg_x = int(MovingAverage)
   
   if KM:
      if abs(kmeanC-MovingAverage) < 45:
         if abs(kmeanC - avg_x) > 15 and numIntersects != 0:
            if abs(kmeanC-MovingAverage) < abs(avg_x-MovingAverage):
               avg_x = (kmeanC + avg_x)//2
         else:
            color = (255, 255, 0)
            avg_x = kmeanC
   #update moving average
   del MA[0]
   MA.append(avg_x)      
   MovingAverage = sum(MA)/len(MA)
   csvlist.append(MovingAverage)
   kmeanC = avg_x
   #this is where the avg circle is drawn
   cv2.circle(frame, (int(MovingAverage), round(height/2)), 10, color, -1)

   #########################################################
   ################DRIVING TIME#############################
   #########################################################
   # Calculate the left and right motor speeds
   winsize = width/numwin
   lAdj = abs(Adj) if Adj < 0 else 0
   rAdj = Adj if Adj > 0 else 0
   lAdj += abs(nearWallAdj) if nearWallAdj < 0 else 0
   rAdj += nearWallAdj if nearWallAdj > 0 else 0
   
   lMotor = 8 + lAdj
   rMotor = 8 + rAdj
   bothAdj = 0

   thisBin = -1
   for i in range(1,numwin+1):
      if MovingAverage < i * winsize:
         thisBin = i-1
         break

   if thisBin == -1:
      print("WOAH NEGATIVE ERROR!!!")
   else:
      if sum(imBin) > 10: #30
         diff = sum(imBin[0:numwin//2]) - sum(imBin[numwin//2+1:numwin])
         if abs(diff) > 2:
            Adj += diff/abs(diff)
         else:
            Adj = 0
         imBin = [0]*numwin
         
      imBin[thisBin] += 1            
      bothAdj = thisBin - numwin//2
      if not bothAdj:
         #lMotor += bothAdj
         #rMotor -= bothAdj

         lMotor = int(clamp(lMotor, 0, 15))
         rMotor = int(clamp(rMotor, 0, 15))
      else:
         if bothAdj < 0:
            lMotor = int(clamp(rMotor/2 + 1,0,15))
            rMotor = int(clamp(rMotor,0,15))
         else:
            lMotor = int(clamp(lMotor,0,15))
            rMotor = int(clamp(lMotor/2 + 1,0,15)) 
   if HOG or HOGOPTICAL:
      if personAvoidance and personAvoidCount == 0:
         personAvoidCount = 1
      if personAvoidCount > 0:
         personAvoidCount -= 1
      #lMotor = lMotor - personAvoidance
      #rMotor = rMotor + personAvoidance
   
   #print(personAvoidCount)
   #print('L:{}, R:{}'.format(lMotor, rMotor))
   if s:
      print("left {}, right {}".format(lMotor, rMotor))
      lMotor = int(lMotor)
      rMotor = int(rMotor)
      ser.write(chr((lMotor << 4 | rMotor)).encode())

   if HOG or HOGOPTICAL:
      for(x1,y1,x2,y2) in pick:
         cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 0),thickness=2)

   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

   if counter <= count:
      pass
      #break

   #display the resulting frame
   cv2.imshow('frame', frame)
   count += 1
   if RECORD:
      out.write(frame)

   #timelist.append(time.time() - start_time)
   #time.sleep(abs(0.0416-delta_time))

if s:
   ser.write(chr(0).encode())
   time.sleep(1)
   ser.close()

if RECORD:
   out.release()

print('~~~~~~END~~~~~~')
print('Frames: {}'.format(count))
print('FPS: {}'.format(int((count)/(time.time() - start_time))))
with open(filename + 'csv', 'r') as f:
   reader = csv.reader(f)
   lines = list(reader)

lines = lines[0][0:count]
csvlist = csvlist[0:count]
#print(lines)
#print(csvlist)

diff = [abs(float(x)/2 - float(y)) for x,y in zip(lines, csvlist)]
#print(diff)
print(len(diff))
print('Avg diff from actual VP: {:.3f}'.format(sum(diff)/len(diff)))
print('Variance of diff from actual VP: {:.3f}'.format(statistics.variance(diff)))
#print('Actual VP Var: {:.3f}'.format(statistics.variance(csvlist)))
#print('Calcul VP Var: {:.3f}'.format(statistics.variance([float(x) for x in lines[0]])))
cap.release()
cv2.destroyAllWindows()
