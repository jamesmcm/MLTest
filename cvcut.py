import cv2
import numpy as np
import sys
import pickle
from scipy.misc import imresize
np.set_printoptions(threshold='nan')
pixelthreshold=0 #for sum method
#dsize=(70,50)
#esize=70*50

dsize=(40,30)
esize=dsize[0]*dsize[1]


def extractdigits(filename):
    a=cv2.imread(filename, 0)
    #print a.shape
    a=a[:,1000:1450]
    #cv2.namedWindow("newtest")
    #cv2.imshow("newtest",a)
    a=cv2.GaussianBlur(a,(3,3), 0)
    orig=a.copy()
    #(thresh, a) = cv2.threshold(a, 80, 255, cv2.THRESH_BINARY)

    #(thresh, a) = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    a=cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 2)

    #a=cv2.bitwise_not(a)

    cv2.imshow("newtest",a)
    cv2.waitKey(0)
    #sys.exit()
    #raw_input()

    b=a.copy()

    contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    mask = np.zeros(a.shape, dtype=np.uint8)
    #cv2.drawContours(mask, contours, 0, 100)
    #cv2.drawContours(mask, contours, 1, 100)
    #cv2.drawContours(mask, contours, 2, 100)
    #cv2.imshow("newtest",mask)
    #cv2.waitKey(0)
    dlist=[]
    output=np.zeros(b.shape,dtype=np.uint8)

    for cont in contours:

        br=cv2.boundingRect(cont)
        #print br
        cv2.rectangle(b, (br[0],br[1]), (br[0]+br[2],br[1]+br[3]), 255)
        charray=np.zeros(dsize, dtype=np.uint8)
        temp=b[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]
        #cv2.imshow("newtest",temp)
        #cv2.waitKey(0)

        if (temp.shape[0]>0.9*dsize[0] and temp.shape[0]<=2*dsize[0] and temp.shape[1]>0.9*dsize[1] and temp.shape[1]<=5*dsize[1]):
            #print temp.shape
            #charray[:temp.shape[0],:temp.shape[1]]=temp
            #(thresh, temp) = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY)
            temp=cv2.bitwise_not(temp)
            temp2=temp.copy()
            cv2.imshow("newtest",temp)
            cv2.waitKey(0)
            #contours2, hierarchy = cv2.findContours(temp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            contours2, hierarchy = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cont2 in contours2:
                br2=cv2.boundingRect(cont2)
                # temp3=temp[br2[1]:br2[1]+br2[3], br2[0]:br2[0]+br2[2]]
                # cv2.imshow("newtest", temp3)
                # print temp3.shape
                # cv2.waitKey(0)


                if br2[3]<dsize[0]+10 and br2[3]>dsize[0]-10 and br2[2]<dsize[1]+10 and br2[2]>dsize[1]-20 and br2[0]>0+(temp.shape[1]/20) and br2[0]<temp.shape[1]-(temp.shape[1]/5):
                    mask = np.zeros(temp2.shape, dtype=np.uint8)
                    cv2.drawContours(mask,[cont2],0,255,-1)
                    temp2=temp.copy()
                    temp2[mask==0]=0

                    temp3=temp2[br2[1]:br2[1]+br2[3], br2[0]:br2[0]+br2[2]]
                    charray=temp3.copy()
                    charray=imresize(charray, dsize)
                    dlist.append(charray)


                    print temp3.shape
                    cv2.imshow("newtest", temp3)
                    cv2.waitKey(0)
            
                #if br2[2]>10 and br2[3]>10:
                    #cv2.rectangle(temp, (br2[0],br2[1]), (br2[0]+br2[2],br2[1]+br2[3]), 255)


            
            # charray=temp.copy()
            # charray=imresize(charray, dsize)
            # dlist.append(charray)
            #output[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]=b[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]
            #cv2.rectangle(output, (br[0],br[1]), (br[0]+br[2],br[1]+br[3]), 255)
        #print temp.shape
        #dlist=dlist[1:-1]
    #cv2.imwrite("readout.png", output)
    #cv2.imshow("newtest",b)
    #cv2.waitKey(0)

    # for item in dlist:  
    #     dlist[0].resize(dsize)
    #     cv2.imshow("newtest",item)
    #     cv2.waitKey(0)
    #     print item.shape
        #raw_input()
    return dlist

# ML part: scikit training
# from sklearn import svm
# import pickle
# #flatten data in to one array with one row per image
# data = np.zeros((10,70*50), dtype=np.uint8)
# for i in range(len(dlist)):
#     data[i]=dlist[i].flatten()

# #print data.shape
# #sols=np.array([5,2,1,0,7,4,3,8,6,9])
# sols=np.array([2,1,0,7,5,4,3,9,8,6])
# classifier=svm.SVC(gamma=0.001, C=10)
# classifier.fit(data, sols)
# print classifier

# f=open("classifier.pkl", "w")
# pickle.dump(classifier, f)
# f.close()


def pmtrain(dlist, sols, classifierfile):
    #Pixel match train:
    #flatten data in to one array with one row per image
    data = np.zeros((len(dlist),esize), dtype=np.uint8)
    for i in range(len(dlist)):
        data[i]=dlist[i].flatten()

    #print data.shape
    #sols=np.array([5,2,1,0,7,4,3,8,6,9])
    #sols=np.array([2,1,0,7,5,4,3,9,8,6])
    tdict={}
    for i in range(len(sols)):
        tdict[sols[i]]=data[i]
    f=open(classifierfile,"w")
    pickle.dump(tdict, f)
    f.close()

def binarise(x):
    if x>pixelthreshold:
        return 1
    else:
        return 0

def sumpmtrain(dlist, sols, classifierfile):
    #Summing pixel match train:
    #Make maximum value=1
    #flatten data in to one array with one row per image
    data = np.zeros((len(dlist),esize), dtype=np.uint8)
    #cv2.namedWindow("newtest2")
    for i in range(len(dlist)):
        data[i]=dlist[i].flatten()
        vecfunc=np.vectorize(binarise)
        data[i]=vecfunc(data[i])
        #test=data[i].copy()
        #test=np.reshape(test,(70,50))
        #cv2.imshow("newtest2",test)
        #cv2.waitKey(0)

    fileexists=True
    try:
        f=open(classifierfile,"r")
        tdict=pickle.load(f)
        f.close()
        
    except:
        tdict={}
        fileexists=False
        for i in range(len(sols)):
            tdict[sols[i]]=data[i]
        
    if fileexists==True:
        for i in range(len(sols)):
            tdict[sols[i]]+=data[i]
            tdict[sols[i]]/=2.0

    f=open(classifierfile,"w")
    pickle.dump(tdict, f)
    f.close()

def sumpmtrain2(dlist, classifierfile):
    #Summing pixel match train:
    #Make maximum value=1
    #flatten data in to one array with one row per image
    fileexists=True
    try:
        f=open(classifierfile,"r")
        tdict=pickle.load(f)
        f.close()
    except:
        tdict={}
        fileexists=False


    for item in dlist:
        data = np.zeros((esize), dtype=np.uint8)
        data=item.flatten()
        vecfunc=np.vectorize(binarise)
        data=vecfunc(data)
        cv2.imshow("newtest2",item)
        ci=cv2.waitKey(0)
        #ci=raw_input()
        ci=ci-48
        if ci in [0,1,2,3,4,5,6,7,8,9]:
            if fileexists==False:
                tdict[ci]=data
        
            elif fileexists==True:
                if ci in tdict.keys():
                    tdict[ci]+=data
                    tdict[ci]/=2.0
                else:
                    tdict[ci]=data

    f=open(classifierfile,"w")
    pickle.dump(tdict, f)
    f.close()


def bool2int(x):
    if x==True:
        return 1
    else:
        return 0

#Pixel match test:
def pmtest(classifierfile, dlist):
    f=open(classifierfile,"r")
    tdict=pickle.load(f)
    f.close()
    data = np.zeros((10,70*50), dtype=np.uint8)
    results=[]
    for i in range(len(dlist)):
        data[i]=dlist[i].flatten()
        resultd=[]
        for j in range(len(tdict.keys())):
            test=(data[i]==tdict[j])
            vecfunc=np.vectorize(bool2int)
            result=np.sum(vecfunc(test))
            resultd.append(float(result)/float(esize))
        print i, resultd
        results.append(resultd.index(max(resultd)))
    print results
    #print data[0]

#Pixel summing test:
#Use logical indices to index where pixels exist
def sumpmtest(classifierfile, dlist):
    f=open(classifierfile,"r")
    tdict=pickle.load(f)
    f.close()
    data = np.zeros((len(dlist),esize), dtype=np.uint8)
    results=[]
    err=[]
    for i in range(len(dlist)):
        data[i]=dlist[i].flatten()
        boolarray=(data[i]>pixelthreshold)
        resultd=[]
        for j in range(len(tdict.keys())):
            result=np.sum(tdict[j][boolarray])
            resultd.append(result/float(esize))
        #print i, resultd
        # sr=reversed(sorted(resultd))
        # srlist=[]
        # for j in sr:
        #     srlist.append(j)
        # err.append(srlist[0]-srlist[1])
        results.append(resultd.index(max(resultd)))
    print results
    #print err
    #print min(err)

    #print data[0]

def sumpmtest2(classifierfile, dlist):
    f=open(classifierfile,"r")
    tdict=pickle.load(f)
    f.close()

    for item in dlist:
        data = np.zeros((esize), dtype=np.uint8)
    #results=[]
    #err=[]
        data=item.flatten()
        boolarray=(data>pixelthreshold)
        resultd=[]
        print tdict.keys()
        for j in range(len(tdict.keys())):
            result=np.sum(tdict[j][boolarray])
            #penalisation factor
            result-=4*np.sum(tdict[j][data<=pixelthreshold])
            resultd.append(result/float(esize))
        print resultd
        # sr=reversed(sorted(resultd))
        # srlist=[]
        # for j in sr:
        #     srlist.append(j)
        # err.append(srlist[0]-srlist[1])
        resultf=(resultd.index(max(resultd)))
        print resultf
        if max(resultd)<-0.1:
            print "IGNORE!"

        print "---"
        cv2.imshow("newtest",item)
        cv2.waitKey(0)


def plotsumpkl(tdictfile):
    f=open(tdictfile,"r")
    tdict=pickle.load(f)
    f.close()
    cv2.namedWindow("newtest2")
    for i in range(10):
        d=tdict[i]*255
        d=np.reshape(d,dsize)
        cv2.imshow("newtest2",d)
        cv2.waitKey(0)




if __name__=="__main__":
    #main here
    # dlist=extractdigits("data/test1.png")
    # sols=np.array([5,2,1,0,7,4,3,8,6,9])
    
    # dlist=extractdigits("data/test2.png")
    # sols=np.array([2,1,0,7,5,4,3,9,8,6])
    # sumpmtrain(dlist, sols, "tdictsum.pkl")
    dlist=extractdigits("data/camtest7.png")
    #sumpmtrain2(dlist, "tdictsumnew.pkl")
    sumpmtest2("tdictsumnew.pkl", dlist)
    #plotsumpkl("tdictsum.pkl")
    #pmtrain(dlist, sols, "tdict.pkl")
    #pmtest("tdict.pkl", dlist)
