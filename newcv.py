import cv2
from numpy import *
from time import sleep
# setup video capture
raw_input("Waiting...")
cap = cv2.VideoCapture(0)
ret,im = cap.read() 
cv2.imwrite("data/test3.png", im)
