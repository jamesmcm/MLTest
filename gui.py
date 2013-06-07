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
        
    def btnclicked(self, widget):
        fdict={"resbtn":self.setResolution()}
        fdict[gtk.Buildable.get_name(widget)]

    def openWindow(self, widget):
        #Make dict of widgets to functions
        fdict={"opencameraconfig":self.openCameraConfig()}
        fdict[gtk.Buildable.get_name(widget)]

    def closeWindow(self,widget):
        fdict={"closecameraconfig":self.applyCameraConfig()}
        fdict[gtk.Buildable.get_name(widget)]

        pass
    def openCameraConfig(self):
        try:
            self.builder.get_object("ccimgbox").remove(self.canvas)
        except:
            pass
        ret,img = self.cap.read() 
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except:
            img=np.zeros((100,100))
        self.builder.get_object("resx").set_text(str(img.shape[1]))
        self.builder.get_object("resy").set_text(str(img.shape[0]))

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
        self.openCameraConfig()



#Main loop:
if __name__ == "__main__":
	mygui = gui()
	mainloop=gtk.main()
