#!/usr/bin/env python
#Imports:
import os
import matplotlib
matplotlib.use('GTK')
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg  
import matplotlib.cm as cm
import numpy as np
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib import lines
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import sys
import math
import datetime
import pickle
import Image #test for PIL
import pygtk
import gtk
import cv2
from scipy.misc import imresize

#Misc functions
def CV_FOURCC(c1, c2, c3, c4) :
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)

def reversetuple((a,b)):
    return (b,a)


#Start class:
class gui:
    """Main application class."""
    def __init__(self):
        self.builder=gtk.Builder()
        self.builder.add_from_file("gladeb.glade")
        myscreen=gtk.gdk.Screen()
        self.screensize=(myscreen.get_width(),myscreen.get_height())
        print self.screensize
        dic={"mainwindowdestroy" : gtk.main_quit,"openwindow":self.openWindow, "closewindow":self.closeWindow, "devicenumchanged":self.changeDeviceNumber, "btnclicked":self.btnclicked}
        self.builder.connect_signals(dic)

        #Initialise defaults
        #Camera config window
        self.devicenumcb = gtk.combo_box_new_text()
        self.builder.get_object("combospace").add(self.devicenumcb)
        self.builder.get_object("combospace").show_all()


        self.devicenumcb.append_text('0')
        self.devicenumcb.append_text('1')
        self.devicenumcb.set_active(0)
        self.devicenumcb.connect("changed", self.changeDeviceNumber)
        self.figure=Figure()
        self.axis=self.figure.add_subplot(111)
        self.cap = cv2.VideoCapture(self.devicenumcb.get_active())

        #Monitor config window
        self.tolerance=25
        self.blocksize=7
        self.d1size=(80,50)
        self.builder.get_object("tolerance").set_text("25")
        self.builder.get_object("blocksize").set_text("7")
        self.builder.get_object("d1x").set_text("50")
        self.builder.get_object("d1y").set_text("80")

    #General functions
    def btnclicked(self, widget):
        call=gtk.Buildable.get_name(widget)
        if call=="resbtn":
            self.setResolution()
        elif call=="imagesavebtn":
            fname=self.builder.get_object("imagesavetext").get_text()
            if fname!="" and fname!=None:
                self.saveImage()
        elif call=="videosavebtn":
            fname=self.builder.get_object("videosavefile").get_text()
            if fname!="" and fname!=None:
                self.saveVideo(fname)

    def openWindow(self, widget):

        #Make dict of widgets to functions
        call=gtk.Buildable.get_name(widget)
        if call=="opencameraconfig":
            self.openCameraConfig()
        elif call=="openimagesave": 
            self.builder.get_object("imagesavewindow").set_visible(1)
        elif call=="openrecordvideo": 
            self.builder.get_object("videosavewindow").set_visible(1)

    def closeWindow(self,widget):
        call=gtk.Buildable.get_name(widget)
        if call=="closecameraconfig":
            self.applyCameraConfig()
        elif call=="imagesaveclosebtn":
            self.builder.get_object("imagesavewindow").set_visible(0)
        elif call=="closevideowindow":
            self.builder.get_object("videosavewindow").set_visible(0)


    #Camera config functions
    def openCameraConfig(self):
        try:
            self.builder.get_object("ccimgbox").remove(self.canvas)
        except:
            pass
        ret,img = self.cap.read() 
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("newtest",img)
        except:
            img=np.zeros((100,100))
        self.builder.get_object("resx").set_text(str(img.shape[1]))
        self.builder.get_object("resy").set_text(str(img.shape[0]))
        self.resolution=(img.shape[1], img.shape[0])

        if img.shape[1]<self.screensize[0] and img.shape[0]<self.screensize[1]:
            pass
        else:
            img=imresize(img, (img.shape[0]/2, img.shape[1]/2))
            

        self.axis.imshow(img, cmap=cm.gray)
        self.canvas=FigureCanvasGTKAgg(self.figure)
        self.canvas.draw()
        self.canvas.show()
        self.builder.get_object("ccimgbox").pack_start(self.canvas, True, True)

        self.builder.get_object("cameraconfig").set_visible(1)
    def applyCameraConfig(self):
        self.builder.get_object("cameraconfig").set_visible(0)
    def changeDeviceNumber(self, widget):
        self.cap = cv2.VideoCapture(self.devicenumcb.get_active())
        self.openCameraConfig()

    def setResolution(self):
        x=self.builder.get_object("resx").get_text()
        y=self.builder.get_object("resy").get_text()
        self.cap.set(3,int(x))
        self.cap.set(4,int(y))
        self.resolution=(int(x),int(y))
        self.openCameraConfig()

    #Image saving
    def saveImage(self,fname):
        if fname!="" or None:
            ret,im=self.cap.read()
            cv2.imwrite(fname,im)

    #Video saving
    def saveVideo(self,fname):
        try:
            if fname.lower()[-4:]!=".avi":
                fname=fname+".avi"
        except:
            fname=fname+".avi"
        video  = cv2.VideoWriter(fname,CV_FOURCC(ord("D"),ord("I"),ord("V"),ord("X")), 25, self.resolution)
        ret,im = self.cap.read() 
        for i in range(75):
            #solve threading problem here
            #Segfaults on opening imshow?
            # get grayscale image
            ret,im = self.cap.read() 
            #im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            video.write(im)
            #cv2.imshow("webcam",im)
            #if (cv2.waitKey(5) != -1):
            #    video.release()
            #    break
        video.release()
        #TODO: May want to chuck away last frame - perhaps do this in analysis


            
#Main loop:
if __name__ == "__main__":
	mygui = gui()
	mainloop=gtk.main()
