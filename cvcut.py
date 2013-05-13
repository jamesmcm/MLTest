import cv2
import numpy as np
import sys
import pickle
from scipy.misc import imresize
a=cv2.imread("data/test1.png", 0)

cv2.namedWindow("newtest")
#cv2.imshow("newtest",a)
a=cv2.GaussianBlur(a,(3,3), 0)

(thresh, a) = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


#a=cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
#a=cv2.bitwise_not(a)

cv2.imshow("newtest",a)
cv2.waitKey(0)
#raw_input()

b=a.copy()

contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
#mask = np.zeros(a.shape, dtype=np.uint8)
#cv2.drawContours(mask, contours, 0, 100)
#cv2.drawContours(mask, contours, 1, 100)
#cv2.drawContours(mask, contours, 2, 100)
dlist=[]
for cont in contours:

    br=cv2.boundingRect(cont)
    #print br
    #cv2.rectangle(b, (br[0],br[1]), (br[0]+br[2],br[1]+br[3]), 255)
    charray=np.zeros((70,50), dtype=np.uint8)
    temp=b[br[1]:br[1]+br[3], br[0]:br[0]+br[2]]
    if (temp.shape[0]>10 and temp.shape[0]<=70 and temp.shape[1]>10 and temp.shape[1]<=50):
        #charray[:temp.shape[0],:temp.shape[1]]=temp
        charray=temp.copy()
        charray=imresize(charray, (70,50))
        dlist.append(charray)
    #print temp.shape
    #dlist=dlist[1:-1]
#cv2.imwrite("chartest1r.png", dlist[0])
#sys.exit()
#for item in dlist:  
    #dlist[0].resize((70,50))
    #cv2.imshow("newtest",item)
    #cv2.waitKey(0)
    #print item.shape
    #raw_input()



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


#Pixel match train:
#flatten data in to one array with one row per image
# data = np.zeros((10,70*50), dtype=np.uint8)
# for i in range(len(dlist)):
#     data[i]=dlist[i].flatten()

# #print data.shape
# #sols=np.array([5,2,1,0,7,4,3,8,6,9])
# sols=np.array([2,1,0,7,5,4,3,9,8,6])
# tdict={}
# for i in range(len(sols)):
#     tdict[sols[i]]=data[i]
# f=open("tdict.pkl","w")
# pickle.dump(tdict, f)
# f.close()



def bool2int(x):
    if x==True:
        return 1
    else:
        return 0

#Pixel match test:
f=open("tdict.pkl","r")
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
        resultd.append(result/3500.0)
    print i, resultd
    results.append(resultd.index(max(resultd)))
print results
