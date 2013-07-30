#!/usr/bin/env python
#Imports:
import os
import matplotlib
matplotlib.use('GTK')
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg  
import matplotlib.cm as cm
import numpy as np
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib import lines
from matplotlib.patches import Rectangle
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

#TODO Fix uneditable text boxes

#Misc functions
def CV_FOURCC(c1, c2, c3, c4) :
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)

def reversetuple((a,b)):
    return (b,a)

class Contour(object):
    def __init__(self, ditem, ritem, label):
        #Set label, position, etc.
        self.ditem=ditem
        self.ritem=ritem
        self.label=label
        

#Start class:
class gui:
    """Main application class."""
    def __init__(self):
        self.builder=gtk.Builder()
        self.builder.add_from_file("gladeb.glade")
        myscreen=gtk.gdk.Screen()
        self.screensize=(myscreen.get_width(),myscreen.get_height())
        print self.screensize
        dic={"mainwindowdestroy" : gtk.main_quit,"openwindow":self.openWindow, "closewindow":self.closeWindow, "devicenumchanged":self.changeDeviceNumber, "btnclicked":self.btnclicked, "fileset":self.fileSet, "mctoolbarbtn":self.mctoolbarClicked, "trkeypress":self.getKeyPress}
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
        self.figurecc=Figure()
        self.axiscc=self.figurecc.add_subplot(111)
        try:
            self.cap = cv2.VideoCapture(self.devicenumcb.get_active())
        except:
            pass

        #Monitor config window
        self.tolerance=25
        self.blocksize=7
        self.d1size=(80,50)
        self.d2size=None
        self.builder.get_object("tolerance").set_text("25")
        self.builder.get_object("blocksize").set_text("7")
        self.builder.get_object("d1x").set_text("50")
        self.builder.get_object("d1y").set_text("80")
        self.figuremc=Figure()
        self.axismc=self.figuremc.add_subplot(111)
        self.mcflipx=False
        self.mcflipy=False
        self.clickstate="none"
        self.crop1=None
        self.crop2=None
        self.builder.get_object("tolerance").set_editable(True)
        self.builder.get_object("blocksize").set_editable(True)
        self.builder.get_object("d1x").set_editable(True)
        self.builder.get_object("d1y").set_editable(True)
        self.builder.get_object("d2x").set_editable(True)
        self.builder.get_object("d2y").set_editable(True)

        self.contours=[]
        self.dlist=[]
        self.trdlist=[]
        self.figuretr=Figure()
        self.axistr=self.figuretr.add_subplot(111)
        self.trframe=0
        self.trtotal=0
        self.traindict={}
        self.solsdict={}

    #General functions
    def fileSet(self, widget):
        call=gtk.Buildable.get_name(widget)
        if call=="monitorconfigloadfile":
            self.loadMonitorImage(widget.get_filename())
        elif call=="trainingfilechooser":
            self.loadTrainingImage(widget.get_filename())

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
        elif call=="setparameters":
            self.setParameters()
        elif call=="setdigits":
            self.setDigitSizes()
        elif call=="mcflipx":
            if self.mcflipx==False:
                self.mcflipx=True
            else:
                self.mcflipx=False
            self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())
        elif call=="mcflipy":
            if self.mcflipy==False:
                self.mcflipy=True
            else:
                self.mcflipy=False
            self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())
        elif call=="addtag":
            self.setClickMode("tag")
        elif call=="cleartags":
            #TODO Clear tags
            pass
        elif call=="addcontour":
            self.setClickMode("addcontour1")
        elif call=="rmcontour":
            self.setClickMode("rmcontour")
        elif call=="splitcontour":
            self.setClickMode("splitcontour")
        elif call=="cropimage":
            self.setClickMode("crop1")
        elif call=="tagokbtn":
            self.curtag=self.builder.get_object("tagname").get_text()   
            self.builder.get_object("tagwindow").set_visible(0)   
            self.contours.append(Contour(self.tempditem,self.tempitem,self.curtag))
        elif call=="tagcancelbtn":
            self.setClickMode("none")
            self.builder.get_object("tagwindow").set_visible(0)
        elif call=="trnext":
            self.trNext()

        elif call=="trprev":
            if self.trframe>0:
                self.trframe-=1
                self.updateTrainingDataWindow()

    def trNext(self):
        if self.trframe<(self.trtotal-1):
            self.trframe+=1    
            self.updateTrainingDataWindow()
        
    def openWindow(self, widget):

        #Make dict of widgets to functions
        call=gtk.Buildable.get_name(widget)
        if call=="opencameraconfig":
            self.openCameraConfig()
        elif call=="openimagesave": 
            self.builder.get_object("imagesavewindow").set_visible(1)
        elif call=="openrecordvideo": 
            self.builder.get_object("videosavewindow").set_visible(1)
        elif call=="openmonitorconfig": 
            self.builder.get_object("monitorconfig").set_visible(1)
        elif call=="opentrainingdatawindow": 
            self.builder.get_object("trainingdatawindow").set_visible(1)


    def closeWindow(self,widget):
        call=gtk.Buildable.get_name(widget)
        if call=="closecameraconfig":
            self.applyCameraConfig()
        elif call=="imagesaveclosebtn":
            self.builder.get_object("imagesavewindow").set_visible(0)
        elif call=="closevideowindow":
            self.builder.get_object("videosavewindow").set_visible(0)
        elif call=="closemonitorconfig":
            self.builder.get_object("monitorconfig").set_visible(0)


    #Camera config functions
    def openCameraConfig(self):
        try:
            self.builder.get_object("ccimgbox").remove(self.canvascc)
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
            

        self.axiscc.imshow(img, cmap=cm.gray) #set scale to 0,255 somehow
        self.canvascc=FigureCanvasGTKAgg(self.figurecc)
        self.canvascc.draw()
        self.canvascc.show()
        self.builder.get_object("ccimgbox").pack_start(self.canvascc, True, True)

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
            ret,im = self.cap.read() 
            video.write(im)
        video.release()
        #TODO: May want to chuck away last frame - perhaps do this in analysis

    #Monitor configuration
    def loadMonitorImage(self,fname):
        if fname[-4:].lower()==".avi":
            #get first frame - look this up
            pass
        elif fname[-4:].lower()==".png" or fname[-4:].lower()==".jpg":
            a=cv2.imread(fname, 0) #note 0 implies grayscale
            #getcontours
        else:
            #Error
            pass
        if self.mcflipx==True and self.mcflipy==True:
            a=cv2.flip(a,-1)
        elif self.mcflipx==True and self.mcflipy==False:
            a=cv2.flip(a,0)
        if self.mcflipy==True and self.mcflipx==False:
            a=cv2.flip(a,1)
        if self.crop1!=None and self.crop2!=None:
            #print str(self.crop1)
            #print str(self.crop2)
            a=a[np.min([self.crop1[1],self.crop2[1]]):np.max([self.crop1[1],self.crop2[1]]),np.min([self.crop1[0],self.crop2[0]]):np.max([self.crop1[0],self.crop2[0]])] 
        (self.monimage,self.dlist,self.rlist)=self.getContours(a,self.d1size)
            #TODO add other digit sizes
        self.drawMonitor()

    def setParameters(self):
        self.tolerance=int(self.builder.get_object("tolerance").get_text())
        self.blocksize=int(self.builder.get_object("blocksize").get_text())
        self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())

    def setDigitSizes(self):
        if (self.builder.get_object("d1y").get_text()!=None and self.builder.get_object("d1y").get_text()!="") and (self.builder.get_object("d1x").get_text()!="" and self.builder.get_object("d1x").get_text()!=None):
            self.d1size=(int(self.builder.get_object("d1y").get_text()),int(self.builder.get_object("d1x").get_text()))
        else:
            self.d1size=None

        if (self.builder.get_object("d2y").get_text()!=None and self.builder.get_object("d2y").get_text()!="") and (self.builder.get_object("d2x").get_text()!="" and self.builder.get_object("d2x").get_text()!=None):
            self.d2size=(int(self.builder.get_object("d2y").get_text()),int(self.builder.get_object("d2x").get_text()))
        else:
            self.d2size=None
            
        #Redo contours, etc.
        self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())

    def getContours(self,a,dsize):
        a=cv2.GaussianBlur(a,(3,3), 0)
        orig=a.copy()
        a=cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.tolerance, self.blocksize)
        b=a.copy()
        contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        mask = np.zeros(a.shape, dtype=np.uint8)
        dlist=[]
        output=np.zeros(b.shape,dtype=np.uint8)
        rlist=[]
        for cont in contours:

            br=cv2.boundingRect(cont)
            charray=np.zeros(dsize, dtype=np.uint8)
            temp=b[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]

            if temp.shape[0]>30 and temp.shape[1]>30:
                temp=cv2.bitwise_not(temp)
                temp2=temp.copy()
                contours2, hierarchy = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for cont2 in contours2:
                    br2=cv2.boundingRect(cont2)

                    if br2[3]<dsize[0]+10 and br2[3]>dsize[0]-10 and br2[2]<dsize[1]+10 and br2[2]>dsize[1]-50 and br2[0]>0+(temp.shape[1]/30) and br2[0]<temp.shape[1]-(temp.shape[1]/5):
                        mask = np.zeros(temp2.shape, dtype=np.uint8)
                        cv2.drawContours(mask,[cont2],0,255,-1)

                        temp2=temp.copy()
                        temp2[mask==0]=0

                        temp3=temp2[br2[1]:br2[1]+br2[3], br2[0]:br2[0]+br2[2]]
                        charray=temp3.copy()
                        charray=imresize(charray, dsize)
                        #dlist.append((charray, br[0]+br2[0], br[1]))

                        if br2[2]>10 and br2[3]>10:
                            #cv2.rectangle(b, (br[0]+br2[0],br[1]+br2[1]), (br[0]+br2[0]+br2[2],br[1]+br2[1]+br2[3]), 100)
                            dlist.append((charray, br[0]+br2[0], br[1]))
                            rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))


        return (b,dlist,rlist)

    def drawMonitor(self):
        try:
            self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
            self.axismc.clear()
            #self.builder.get_object("mctoolbar").remove(self.mctoolbar)	
        except:
            pass

        #Add cropping
        self.axismc.imshow(self.monimage, cmap=cm.gray) #set scale to 0,255 somehow

        #Maybe this needn't be redefined for every draw - only need draw() but not drawn often anyway
        self.canvasmc=FigureCanvasGTKAgg(self.figuremc)

        self.canvasmc.draw()
        self.canvasmc.show()
        self.canvasmc.mpl_connect('motion_notify_event', self.mcHoverOnImage)
        self.canvasmc.mpl_connect('button_release_event', self.mcCaptureClick)

        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)
        for item in self.rlist:
            #Structure of rlist:
            #rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))
            r=Rectangle(item[0], item[1], item[2], fill=False, color="red")
            #Rectangle has (lowerleft, width, height)
            self.axismc.add_patch(r)
        

    def mcHoverOnImage(self, event):
        if event.x!=None and event.y!=None and event.xdata!=None and event.ydata!=None:
            pass
        
    def mcCaptureClick(self, event):
        #print "click"
        if self.clickstate=="none":
            pass
        #elif not(event.x==None or event.y==None or event.xdata==None or event.ydata==None):
        else:
            if self.clickstate=="crop1":
                self.crop1=(int(round(event.xdata)), int(round(event.ydata)))
                self.setClickMode("crop2")
            elif self.clickstate=="crop2":
                self.crop2=(int(round(event.xdata)), int(round(event.ydata)))
                self.setClickMode("none")
                self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())    
            elif self.clickstate=="tag":
                #Check if we are inside contour, if so present label window, if not, ignore
                #Contours checked by rlist?
                coords=(int(round(event.xdata)), int(round(event.ydata)))
                found=False
                for i in range(len(self.rlist)):
                    item=self.rlist[i]
                    if (coords[0] >= item[0][0]) and (coords[0] <= (item[0][0]+item[1])) and (coords[1] >= item[0][1]) and (coords[1] <= item[0][1]+item[2]):
                        #Found contour, create contour object for final contour list
                        found=True
                        break
                if found==True:

                    self.tempitem=item
                    self.tempditem=self.rlist[i]
                    self.builder.get_object("tagwindow").set_visible(1)
                        #self.contours.append(Contour(item,self.curtag))
                        

    def mctoolbarClicked(self,widget):
        call=gtk.Buildable.get_name(widget)
        if call=="mczoomin":
            self.mcZoomIn()
        elif call=="mczoomout":
            self.mcZoomOut()
        elif call=="mcpanleft":
            self.mcPanLeft()
        elif call=="mcpanright":
            self.mcPanRight()
        elif call=="mcpanup":
            self.mcPanUp()
        elif call=="mcpandown":
            self.mcPanDown()
        elif call=="mcresetzoom":
            self.mcResetZoom()


    def mcZoomIn(self):
        xlims=self.axismc.get_xlim()
        ylims=self.axismc.get_ylim()
        xchange=abs(xlims[1]-xlims[0])*0.1
        ychange=abs(ylims[1]-ylims[0])*0.1
        self.axismc.set_xlim(left=xlims[0]+xchange, right=xlims[1]-xchange)
        self.axismc.set_ylim(top=ylims[1]+ychange, bottom=ylims[0]-ychange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)
        
    def mcZoomOut(self):
        xlims=self.axismc.get_xlim()
        ylims=self.axismc.get_ylim()
        xchange=abs(xlims[1]-xlims[0])*0.111
        ychange=abs(ylims[1]-ylims[0])*0.111
        self.axismc.set_xlim(left=xlims[0]-xchange, right=xlims[1]+xchange)
        self.axismc.set_ylim(top=ylims[1]-ychange, bottom=ylims[0]+ychange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)

    def mcPanLeft(self):
        xlims=self.axismc.get_xlim()
        xchange=abs(xlims[1]-xlims[0])*0.1
        self.axismc.set_xlim(left=xlims[0]-xchange, right=xlims[1]-xchange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)

    def mcPanRight(self):
        xlims=self.axismc.get_xlim()
        xchange=abs(xlims[1]-xlims[0])*0.1
        self.axismc.set_xlim(left=xlims[0]+xchange, right=xlims[1]+xchange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)

    def mcPanDown(self):
        ylims=self.axismc.get_ylim()
        ychange=abs(ylims[1]-ylims[0])*0.1
        self.axismc.set_ylim(top=ylims[1]+ychange, bottom=ylims[0]+ychange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)
	    
    def mcPanUp(self):
        ylims=self.axismc.get_ylim()
        ychange=abs(ylims[1]-ylims[0])*0.1
        self.axismc.set_ylim(top=ylims[1]-ychange, bottom=ylims[0]-ychange)
        self.builder.get_object("monitorconfigspace").remove(self.canvasmc)
        self.builder.get_object("monitorconfigspace").pack_start(self.canvasmc, True, True)

    def mcResetZoom(self):
        #Reset view to original somehow - fit entire image
        pass

    def setClickMode(self,mode):
        #none, crop1, crop2, tag, addcontour1, addcontour2, rmcontour, splitcontour
        self.builder.get_object("mcclickmode").set_label("Mode:" + str(mode))
        self.clickstate=mode

    def loadTrainingImage(self,fname):
        if fname[-4:].lower()==".avi":
            #get first frame - look this up
            pass
        elif fname[-4:].lower()==".png" or fname[-4:].lower()==".jpg":
            a=cv2.imread(fname, 0) #note 0 implies grayscale
            #getcontours
        else:
            #Error
            pass
        if self.mcflipx==True and self.mcflipy==True:
            a=cv2.flip(a,-1)
        elif self.mcflipx==True and self.mcflipy==False:
            a=cv2.flip(a,0)
        if self.mcflipy==True and self.mcflipx==False:
            a=cv2.flip(a,1)
        if self.crop1!=None and self.crop2!=None:
            #print str(self.crop1)
            #print str(self.crop2)
            a=a[np.min([self.crop1[1],self.crop2[1]]):np.max([self.crop1[1],self.crop2[1]]),np.min([self.crop1[0],self.crop2[0]]):np.max([self.crop1[0],self.crop2[0]])] 
        (self.monimage,self.dlist,self.rlist)=self.getContours(a,self.d1size)

        #for cont in self.contours:
            #add individual digits to list, then tag list
            #self.dlist.append(self.monimage[cont.ritem[0][1]:cont.ritem[0][1]+cont.ritem[2],cont.ritem[0][0]:cont.ritem[0][0]+cont.ritem[1]])

        #TODO FIX THIS - need to take ditem of new image, not config one, where the coords are the same
        #for cont in self.contours:
            #self.dlist.append(cont.ditem[0])
        for i in range(len(self.rlist)):
            for cont in self.contours:
                #if self.rlist[i]==cont.ritem:
                if np.abs(self.rlist[i][0][0]-cont.ritem[0][0])<=4 and np.abs(self.rlist[i][0][1]-cont.ritem[0][1])<=4:
                    self.trdlist.append(self.dlist[i])
                    self.trtotal+=1
                    #could add width, height check as well

        #update display
        self.updateTrainingDataWindow()

    def updateTrainingDataWindow(self):
        #Use curframe number, like TesiDogs program
        try:
            self.builder.get_object("bvbox3").remove(self.canvastr)
            self.axistr.clear()
        except:
            pass

        self.axistr.imshow(self.trdlist[self.trframe][0], cmap=cm.gray) #set scale to 0,255 somehow
        self.canvastr=FigureCanvasGTKAgg(self.figuretr)
        self.canvastr.draw()
        self.canvastr.show()
        self.builder.get_object("bvbox3").pack_start(self.canvastr, True, True)
        self.builder.get_object("trframecount").set_label(str(self.trframe+1) + "/" + str(self.trtotal))
        #self.builder.get_object("trcursol").set_label(str(self.trframe+1) + "/" + str(self.trtotal))
        
        #TODO update labels
        #bvbox3
        #trframecount
        #trcursol

    def getKeyPress(self, widget, event):
        #TODO training
        #GTKwidget keyreleaseevent
        ci=event.keyval
        ci=ci-48
        data=self.trdlist[self.trframe][0]
        if event.keyval==45:
            #set to not a number
            self.trNext()
        elif ci in [0,1,2,3,4,5,6,7,8,9]:
            if ci in self.traindict.keys():
                self.traindict[ci]+=data
                self.traindict[ci]/=2.0
                self.trNext()
            else:
                self.traindict[ci]=data
                self.trNext()

#Main loop:
if __name__ == "__main__":
	mygui = gui()
	mainloop=gtk.main()
