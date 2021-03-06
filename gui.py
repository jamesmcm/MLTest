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


#TODO Add classifier and output
#TODO Add ability to save classifier pickle
#TODO Add second digit size

#TODO copy crop function for setting digit size
# TODO finish digit classification

#Misc functions
def CV_FOURCC(c1, c2, c3, c4) :
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)

def reversetuple((a,b)):
    return (b,a)

class Contour(object):
    def __init__(self, ditem, ritem, label):
        #Set label, position, etc.
        #redundant use of memory
        self.ditem=ditem
        self.ritem=ritem
        self.label=label
        self.digit=None
        

#Start class:
class gui:
    """Main application class."""
    def __init__(self):
        self.builder=gtk.Builder()
        self.dsizes=[]
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

        self.pixelthreshold=0

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
        self.figuretmc=Figure()
        self.axistmc=self.figuretmc.add_subplot(111)
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
        elif call=="testmcfile":
            self.loadMonitorImage(widget.get_filename(), testmc=True)

    def btnclicked(self, widget):
        call=gtk.Buildable.get_name(widget)
        if call=="resbtn":
            self.setResolution()
        elif call=="imagesavebtn":
            fname=self.builder.get_object("imagesavetext").get_text()
            if fname!="" and fname!=None:
                self.saveImage(fname)
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
        elif call=="getdigitsize":
            self.setClickMode("getdigit1")
        elif call=="tagokbtn":
            self.curtag=self.builder.get_object("tagname").get_text()   
            self.builder.get_object("tagwindow").set_visible(0)   
            self.contours.append(Contour(self.tempditem,self.tempitem,self.curtag))
            if not (self.d1size in self.dsizes):
                self.dsizes.append(self.d1size)
        elif call=="tagcancelbtn":
            self.setClickMode("none")
            self.builder.get_object("tagwindow").set_visible(0)
        elif call=="trnext":
            self.trNext()

        elif call=="trprev":
            if self.trframe>0:
                self.trframe-=1
                self.updateTrainingDataWindow()
        elif call=="savetrdata":
            fn=self.builder.get_object("trfile").get_text()
            f=open(fn, "w")
            pickle.dump(self.traindict, f)
            f.close()
        elif call=="allconts":
            #show all contours in monitor config window
            self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename(), allconts=True)
            
        elif call=="clearunsaved":
            #redraw only those ritems in self.contours
            self.drawMonitor(clearunsaved=True)
            
        elif call=="liverecord":
            #TODO test? Fix all parameters here
            #start loop to get data
            fn=self.builder.get_object("livefile").get_text()
            f=open(fn,"r")
            #log to f

            while True:
                self.livelist=[]
                #get image using current camera config
                ret,im=self.cap.read()
                #run contour detection
                (self.monimage,self.dlist,self.rlist)=self.getContours(a,self.d1size)
                #take only labelled ones
                for i in range(len(self.rlist)):
                    for j in range(len(self.contours)):
                        #if self.rlist[i]==cont.ritem:
                        #TODO remove hidden parameters here
                        cont=self.contours[j]
                        if np.abs(self.rlist[i][0][0]-cont.ritem[0][0])<=4 and np.abs(self.rlist[i][0][1]-cont.ritem[0][1])<=4:
                            #Need to append x position, label
                            self.livelist.append((self.dlist[i],j))
                    #could add width, height check as well
                
                #run digit analysis
                #TODO - modify this for known number of digits?
                for item2 in self.livelist:
                    item=item2[0][0]
                    q=False
                    #data = np.zeros((esize), dtype=np.uint8)
                    esize1=self.d1size[0]*self.d1size[1]
                #results=[]
                #err=[]
                    data=item.flatten()
                    boolarray=(data>self.pixelthreshold)
                    resultd=[]
                    #print self.traindict.keys()
                    for j in range(len(self.traindict.keys())):
                        result=np.sum(self.traindict[j][boolarray])
                        #penalisation factor
                        result-=4*np.sum(self.traindict[j][data<=self.pixelthreshold])
                        resultd.append(result/float(esize1))
                    #print resultd
                    # sr=reversed(sorted(resultd))
                    # srlist=[]
                    # for j in sr:
                    #     srlist.append(j)
                    # err.append(srlist[0]-srlist[1])
                    resultf=(resultd.index(max(resultd)))
                    #print resultf
                    if max(resultd)<-0.1:
                        #print "IGNORE!"
                        q=True

                    #print "---"
                    #cv2.imshow("newtest",item)
                    #cv2.waitKey(0)
                    #Append digit to correct place
                    #rl = {mlabel:{x:(1,q)}}

                    #use label instead of y co-ordinate as constant
                    if self.contours[item2[1]].label in rl.keys():
                        rl[self.contours[item2[1]].label][item2[0][1]]=(resultf, q)
                    else:
                        rl[self.contours[item2[1]].label]={item2[0][1]:(resultf,q)}

                #Want data structure instead of string
                
                for key in sorted(rl.iterkeys()):
                    #print "%s: %s" % (key, mydict[key]) 

                    for key2 in sorted(rl[key].iterkeys()):
                        string+=str(rl[key][key2][0])
                        #if rl[key][key2][1]==False:
                            #create solutions dictionary
                            #string+=str(rl[key][key2][0])
                            
                        #if rl[key][key2][1]==True:
                            #string+="?"
                    solsdict[self.contours[item2[1]].label]=string
                #reconstruct labelled data
                for item in solsdict.keys():
                    f.write(item +":" + solsdict[item])
                f.write("\n")
                #log to f


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
        elif call=="openlive": 
            self.builder.get_object("livewindow").set_visible(1)
        elif call=="opentmc":
            self.builder.get_object("testmcwindow").set_visible(1)

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
        elif call=="closetrainingwindow":
            self.builder.get_object("trainingdatawindow").set_visible(0)
        elif call=="closelive":
            self.builder.get_object("livewindow").set_visible(0)
        elif call=="closetc":
            self.builder.get_object("testmcwindow").set_visible(0)
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
        for i in range(1000):
            ret,im = self.cap.read() 
            video.write(im)
        video.release()
        #TODO: May want to chuck away last frame - perhaps do this in analysis

    #Monitor configuration
    def loadMonitorImage(self,fname, allconts=False, testmc=False):
        if fname[-4:].lower()==".avi":
            #TODO get first frame - look this up
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

            #TODO add other digit sizes
        if testmc==True:
            self.trlist=[]
            for dsize in self.dsizes:
                (self.tmonimage,self.dlist,rlist)=self.getContours(a,dsize)
                self.trlist+=rlist
            self.drawMonitorTest()
        else:
            if allconts==False:
                (self.monimage,self.dlist,self.rlist)=self.getContours(a,self.d1size)
                self.drawMonitor()
            else:
                (self.monimage,self.rlist1,self.rlist2)=self.getAllContours(a)
                self.drawMonitor(allconts=True)

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

            if temp.shape[0]>10 and temp.shape[1]>10:
                temp=cv2.bitwise_not(temp)
                temp2=temp.copy()
                contours2, hierarchy = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for cont2 in contours2:
                    br2=cv2.boundingRect(cont2)
                    #important hidden parameters
                    if br2[3]<dsize[0]+20 and br2[3]>dsize[0]-20 and br2[2]<dsize[1]+20 and br2[2]>dsize[1]-60:
                        #After cropping, edge constrains not necessary
                        # and br2[0]>0+(temp.shape[1]/40.0) and br2[0]<temp.shape[1]-(temp.shape[1]/40.0)
                        mask = np.zeros(temp2.shape, dtype=np.uint8)
                        cv2.drawContours(mask,[cont2],0,255,-1)

                        temp2=temp.copy()
                        temp2[mask==0]=0

                        temp3=temp2[br2[1]:br2[1]+br2[3], br2[0]:br2[0]+br2[2]]
                        charray=temp3.copy()
                        charray=imresize(charray, dsize)
                        #dlist.append((charray, br[0]+br2[0], br[1]))

                        if br2[2]>5 and br2[3]>5:
                            #cv2.rectangle(b, (br[0]+br2[0],br[1]+br2[1]), (br[0]+br2[0]+br2[2],br[1]+br2[1]+br2[3]), 100)
                            dlist.append((charray, br[0]+br2[0], br[1]))
                            rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))


        return (b,dlist,rlist)

    def getAllContours(self,a):
        a=cv2.GaussianBlur(a,(3,3), 0)
        orig=a.copy()
        a=cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.tolerance, self.blocksize)
        b=a.copy()
        contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        mask = np.zeros(a.shape, dtype=np.uint8)

        output=np.zeros(b.shape,dtype=np.uint8)
        rlist1=[]
        rlist2=[]
        for cont in contours:

            br=cv2.boundingRect(cont)
            rlist1.append(((br[0], br[1]), br[2], br[3]))
            temp=b[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]

            if temp.shape[0]>5 and temp.shape[1]>5:
                temp=cv2.bitwise_not(temp)
                temp2=temp.copy()
                contours2, hierarchy = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                for cont2 in contours2:
                    br2=cv2.boundingRect(cont2)

                    #if br2[3]<dsize[0]+10 and br2[3]>dsize[0]-10 and br2[2]<dsize[1]+10 and br2[2]>dsize[1]-50 and br2[0]>0+(temp.shape[1]/30) and br2[0]<temp.shape[1]-(temp.shape[1]/5):
                    mask = np.zeros(temp2.shape, dtype=np.uint8)
                    cv2.drawContours(mask,[cont2],0,255,-1)

                    temp2=temp.copy()
                    temp2[mask==0]=0

                    temp3=temp2[br2[1]:br2[1]+br2[3], br2[0]:br2[0]+br2[2]]
                    #dlist.append((charray, br[0]+br2[0], br[1]))

                    if br2[2]>3 and br2[3]>3:
                        #cv2.rectangle(b, (br[0]+br2[0],br[1]+br2[1]), (br[0]+br2[0]+br2[2],br[1]+br2[1]+br2[3]), 100)
                        #dlist.append((charray, br[0]+br2[0], br[1]))
                        rlist2.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))


        return (b,rlist1, rlist2)


    def drawMonitor(self, allconts=False, clearunsaved=False):
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

        #TODO stop this getting so complicated
        if clearunsaved==False:
            if allconts==False:
                for item in self.rlist:
                    #Structure of rlist:
                    #rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))
                    r=Rectangle(item[0], item[1], item[2], fill=False, color="red")
                    #Rectangle has (lowerleft, width, height)
                    self.axismc.add_patch(r)               
            elif allconts==True:
                #allcontours
                for item in self.rlist1:
                    #Structure of rlist:
                    #rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))
                    r=Rectangle(item[0], item[1], item[2], fill=False, color="blue")
                    #Rectangle has (lowerleft, width, height)
                    self.axismc.add_patch(r)
                for item in self.rlist2:
                    #Structure of rlist:
                    #rlist.append(((br[0]+br2[0], br[1]+br2[1]), br2[2], br2[3]))
                    r=Rectangle(item[0], item[1], item[2], fill=False, color="green")
                    #Rectangle has (lowerleft, width, height)
                    self.axismc.add_patch(r)

        #Always draw saved contours in blue
        for ditem in self.contours:
            item=ditem.ritem
            r=Rectangle(item[0], item[1], item[2], fill=False, color="blue")
            self.axismc.add_patch(r)


    def drawMonitorTest(self):
        try:
            self.builder.get_object("tmcspace").remove(self.canvastmc)
            self.axistmc.clear()
        except:
            pass
        #Add cropping
        self.axistmc.imshow(self.tmonimage, cmap=cm.gray) #set scale to 0,255 somehow

        #Maybe this needn't be redefined for every draw - only need draw() but not drawn often anyway
        self.canvastmc=FigureCanvasGTKAgg(self.figuretmc)

        self.canvastmc.draw()
        self.canvastmc.show()
        self.builder.get_object("tmcspace").pack_start(self.canvastmc, True, True)


        for i in range(len(self.trlist)):
            for cont in self.contours:
                #if self.rlist[i]==cont.ritem:
                #TODO remove hidden parameters here
                if np.abs(self.trlist[i][0][0]-cont.ritem[0][0])<=4 and np.abs(self.trlist[i][0][1]-cont.ritem[0][1])<=4:
                    item=self.trlist[i]
                    r=Rectangle(item[0], item[1], item[2], fill=False, color="blue")
                    self.axistmc.add_patch(r)
                    #could add width, height check as well

        #Always draw saved contours in blue
        for ditem in self.contours:
            item=ditem.ritem



    def mcHoverOnImage(self, event):
        #add contour stuff here if not too expensive
        #find innermost contour
        #Cannot afford to redraw, must work out how to remove rectangle afterwards since only one at a time
        #TODO
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

            elif self.clickstate=="getdigit1":
                self.getdigit1=(int(round(event.xdata)), int(round(event.ydata)))
                self.setClickMode("getdigit2")

            elif self.clickstate=="crop2":
                self.crop2=(int(round(event.xdata)), int(round(event.ydata)))
                self.setClickMode("none")
                self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())    

            elif self.clickstate=="getdigit2":
                self.getdigit2=(int(round(event.xdata)), int(round(event.ydata)))
                #apply stuff
                self.d1size=(np.abs(self.getdigit2[1]-self.getdigit1[1]),np.abs(self.getdigit2[0]-self.getdigit1[0]))
                self.builder.get_object("d1x").set_text(str(np.abs(self.getdigit2[0]-self.getdigit1[0])))
                self.builder.get_object("d1y").set_text(str(np.abs(self.getdigit2[1]-self.getdigit1[1])))
                self.setClickMode("none")
                self.loadMonitorImage(self.builder.get_object("monitorconfigloadfile").get_filename())


            elif self.clickstate=="tag":
                #Check if we are inside contour, if so present label window, if not, ignore
                #Contours checked by rlist?
                coords=(int(round(event.xdata)), int(round(event.ydata)))
                #found=False
                #Find innermost not just first contour
                fitem=None
                fi=None
                for i in range(len(self.rlist)):
                    item=self.rlist[i]
                    if (coords[0] >= item[0][0]) and (coords[0] <= (item[0][0]+item[1])) and (coords[1] >= item[0][1]) and (coords[1] <= item[0][1]+item[2]):
                        #Found contour, create contour object for final contour list
                        if fitem==None:
                            fitem=item
                            fi=i
                        else:
                            if (item[0][0] >= fitem[0][0]) and (item[0][0]+item[1] <= (fitem[0][0]+fitem[1])) and (item[0][1] >= fitem[0][1]) and (item[0][1]+item[2] <= fitem[0][1]+fitem[2]):
                                fitem=item
                                fi=i
                if fitem!=None:
                    self.tempitem=fitem
                    self.tempditem=self.rlist[fi]
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
        #none, crop1, crop2, tag, addcontour1, addcontour2, rmcontour, splitcontour, getdigit1, getdigit2
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
                #TODO remove hidden parameters here
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
        if event.keyval==45:
            #set to not a number
            self.trNext()
        elif ci in [0,1,2,3,4,5,6,7,8,9]:
            data=self.trdlist[self.trframe][0].flatten()
            if ci in self.traindict.keys():
                self.traindict[ci]+=data
                self.traindict[ci]/=2.0
                self.trNext()
            else:
                self.traindict[ci]=data
                self.trNext()

    def sumpmtestlive(self):
        rl={}

        for item2 in dlist:
            item=item2[0]
            q=False
            data = np.zeros((esize1), dtype=np.uint8)
        #results=[]
        #err=[]
            data=item.flatten()
            boolarray=(data>self.pixelthreshold)
            resultd=[]
            #print tdict.keys()
            for j in range(len(tdict.keys())):
                result=np.sum(tdict[j][boolarray])
                #penalisation factor
                result-=4*np.sum(tdict[j][data<=self.pixelthreshold])
                resultd.append(result/float(esize1))
            #print resultd
            # sr=reversed(sorted(resultd))
            # srlist=[]
            # for j in sr:
            #     srlist.append(j)
            # err.append(srlist[0]-srlist[1])
            resultf=(resultd.index(max(resultd)))
            #print resultf
            if max(resultd)<-0.1:
                #print "IGNORE!"
                q=True

            #print "---"
            #cv2.imshow("newtest",item)
            #cv2.waitKey(0)
            #Append digit to correct place
            #rl = {y:{x:(1,q)}}

            if item2[2] in rl.keys():
                rl[item2[2]][item2[1]]=(resultf, q)
            else:
                rl[item2[2]]={item2[1]:(resultf,q)}

        string=""
        for key in sorted(rl.iterkeys()):
            #print "%s: %s" % (key, mydict[key]) 

            for key2 in sorted(rl[key].iterkeys()):
                if rl[key][key2][1]==False:
                    string+=str(rl[key][key2][0])
                #if rl[key][key2][1]==True:
                    #string+="?"
            string+=" "

        print string


#Main loop:
if __name__ == "__main__":
	mygui = gui()
	mainloop=gtk.main()
