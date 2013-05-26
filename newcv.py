import cv2
from numpy import *
from time import sleep
# setup video capture
raw_input("Waiting...")
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
ret,im = cap.read() 
cv2.imwrite("camtest.png", im)
