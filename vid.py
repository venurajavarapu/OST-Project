import numpy as np
import cv2
from numpy.linalg import norm
import lasttrail as t3
    
bin_n=16
font=cv2.FONT_HERSHEY_SIMPLEX
prev = None
text=""
count=0
camera=int(raw_input("Camera to be used --- Enter 0 or 1 : "))
cap=cv2.VideoCapture(camera)
while(True):
    #print "cap read"
    ret,frame=cap.read()
    #print "frame read"
    #print "draw rect"
    cv2.rectangle(frame,(384,100),(560,278),(0,255,0),3)
    image=frame[100:328,384:560]
    #print "img caped"
    #print "writing captured image"
    cv2.imwrite('capImage.jpg',image)
    #print "image written and reading started"
    img=cv2.imread('capImage.jpg')
    #print "image read in vid and model called"
    ch=t3.modelm()
    if ch==prev:
        prev=ch
        count+=1
    else:
        prev=ch
        count=0
    if count==30:
        if ch=="Q":
            ch=" "
            text=text+ch
        elif ch=="K":
            ch=""
            if text!="":
                text=text[:-1]
        else:
            text=text+ch
    cv2.putText(frame,text,(10,400),font,2,(255,0,0),2)
    #print "model ended and returned to vid"
    cv2.imshow('actual',img)
    #print "image displayed"
    cv2.putText(frame,ch,(10,200),font,3,(0,255,0),2)
    #print "frame to be displayed"
    cv2.imshow('frame',frame)
    #print "frame displayed"     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
