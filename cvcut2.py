import cv2
import numpy as np

a=cv2.imread("test1.png", 0)

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
        charray[:temp.shape[0],:temp.shape[1]]=temp
        dlist.append(charray)
    #print temp.shape
    #dlist=dlist[1:-1]

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
# sols=np.array([5,2,1,0,7,4,3,8,6,9])
# classifier=svm.SVC(gamma=0.001)
# classifier.fit(data, sols)

# f=open("classifier.pkl", "w")
# pickle.dump(classifier, f)
# f.close()

#ML: testing
from sklearn import svm
import pickle
f=open("classifier.pkl", "r")
classifier=pickle.load(f)
f.close()
data = np.zeros((10,70*50), dtype=np.uint8)
for i in range(len(dlist)):
    data[i]=dlist[i].flatten()
output=classifier.predict(data)
print output
