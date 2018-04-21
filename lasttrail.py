import numpy as np
import cv2
from numpy.linalg import norm
from sklearn import svm
#print "trail3 start"
bin_n=16
svm_parameters=dict(kernel_type=cv2.SVM_RBF,svm_type=cv2.SVM_C_SVC,c=2.67,gamma=5.383)
#svm_parameters=dict(kernel_type=cv2.SVM_LINEAR,svm_type=cv2.SVM_C_SVC,c=2.67,gamma=5.383)
font=cv2.FONT_HERSHEY_SIMPLEX
#fgbg=cv2.createBackgroundSubtractorMOG()
label=None
previouslabel=None
output=" "
previousoutput=" "
count=0
def hog(images):
    sam=[]
    for img in images:
        bin_n=16
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0,3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1,3)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi)) # quantizing binvalues in (0...16)
        bin_cells = bins[:100,:100], bins[100:,:100], bins[:100,100:], bins[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists) # hist is a 64 bit vector

        eps = 1e-7
        hist=hist/(hist.sum()+eps)
        hist=np.sqrt(hist)
        hist=hist/(norm(hist)+eps)
        
        sam.append(hist)
    return np.float32(sam)
#train   model
num=17
images=[]
samples=[]
for i in range(65,num+65):
    for j in range(1,401):
        print ("loading "+str(unichr(i))+" "+str(j))
        images.append(cv2.imread('TrainData/'+unichr(i)+'_'+str(j)+'.jpg',0))
labels=np.repeat(np.arange(1,num+1),400)

samples = hog(images)

#svm=cv2.SVM(C=2.67,gamma=5.383)
#mysvm=cv2.SVM()
#print samples
print labels

clf=svm.SVC()
clf.fit(samples,labels)

#print mysvm.train(samples,labels,params=svm_parameters)
#print "svm trained"
#gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
def modelm():
    #print "started"
    
    lower=np.array([0,48,80],dtype="uint8")
    upper=np.array([20,255,255],dtype="uint8")
    #lower=np.array([0,10,60],dtype="uint8")
    #upper=np.array([20,150,255],dtype="uint8")
    #cv2.imshow("frame",gray)

    #cv2.rectangle(frame,(384,100),(510,328),(0,255,0),3)
    #image=frame[100:328,384:510]
    #image=cv2.imread('TrainData/G_12.jpg')
    image=cv2.imread('capImage.jpg')

    cv2.imshow('actual',image)

    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    hsvmask=cv2.inRange(hsv,lower,upper)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    hsvmask=cv2.erode(hsvmask,kernel,iterations=2)
    hsvmask=cv2.dilate(hsvmask,kernel,iterations=2)
    
    hsvmask=cv2.GaussianBlur(hsvmask,(3,3),0)
    skinsegment=cv2.bitwise_and(image,image,mask=hsvmask)



    
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))
    ycrcbmask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
    
    #fgmask=fgbg.apply(frame)
    #print "finding contours"
    contours,hierarchy=cv2.findContours(ycrcbmask.copy(),cv2.RETR_EXTERNAL,2)
    #print "contour found started checking max contour"
    minarea=2000
    maxc=None
    maxarea=minarea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxarea):
            maxarea=area
            maxc=cnt
    cnt=maxc
    image1=image
    #print "max contour found"
    #print cnt

    a=384
    b=100
    c=176
    d=178
    #if cnt.any()!=None:
    if maxarea!=2000:
        a,b,c,d=cv2.boundingRect(cnt)
        cv2.rectangle(image,(a,b),((a+c),(b+d)),(0,255,0),2)
        image1=image[b:b+d,a:a+c]
    #image1=cv2.bitwise_and(image1,image1,mask=hsvmask[b:b+d,a:a+c])
    image1=cv2.bitwise_and(image1,image1,mask=ycrcbmask[b:b+d,a:a+c])
    image1=cv2.resize(image1,(200,200))
    finalimage=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    #predict model util
    #ans=train.predict(model,finalimage)
    timg=[]
    timg.append(finalimage)
    #print "calling hog"
    sample1=hog(timg)
    #print "hog ended"
    #print samples[0]
    #print sample1
    sample=sample1[0]
    #print sample
    #svm=cv2.SVM(C=2.67,gamma=5.383)
    #mysvm=svm.SVC()
    ans1=[]
    #ans1=svm.predict_all(sample).ravel()
    #ans1=mysvm.predict(sample).ravel()
    #print "calling predict"
    ans1=clf.predict(sample1)
    #print "predicted"
    ans=ans1[0]
    print ans
    img=cv2.imread("TrainData/"+(unichr(int(ans1[0]+64))+"_1.jpg"))
    label=unichr(int(ans1[0]+64))
    if label!=None:
        if previouslabel==label:
            count+=1
        else :
            count=0
    #print label
    #cv2.putText(frame,label,(50,140),font,8,(0,125,155),2)
    
    cv2.imshow('predicted',img)
    cv2.imshow('actual gray',ycrcbmask)
    #cv2.imshow('sctual hsv',hsvmask)
    cv2.imshow('actual',image1)
    cv2.imshow('actual final',finalimage)
    #print "ended"
    return unichr(int(ans1[0]+64))
    #cv2.putText(frame,label,(10,500),font,4,(255,0,0),2)
    #cv2.imshow("Frame",frame)
    #cv2.imshow("frame",skinsegment)
    #cv2.imshow("frame",ycrcbmask)
    #cv2.putText(ycrcbmask,"Hello World",(10,500),font,4,(255,0,0),2)
    #k=cv2.waitKey(5000) & 0xFF
    #if k==27:
    #    break

