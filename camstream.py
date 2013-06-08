import cv2
from scipy.misc import imresize

def CV_FOURCC(c1, c2, c3, c4) :
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)

cap = cv2.VideoCapture(1)
cap.set(3,1920)
cap.set(4,1080)
#video  = cv2.VideoWriter('video.avi',CV_FOURCC(ord("D"),ord("I"),ord("V"),ord("X")), 25, (1920, 1080));
ret,im = cap.read() 
#prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
while True:
    # get grayscale image
    ret,im = cap.read() 
    #video.write(im)
    #cv2.imshow("webcam",im)
    #if (cv2.waitKey(5) != -1):
    #    break
    
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray=imresize(gray,(270,480))
    #gray=gray[:800,1000:1450]
    # compute flow
    # plot the flow vectors
    cv2.imshow('Optical flow',gray) 
    if cv2.waitKey(5) == 27:
        break

#video.release()  




