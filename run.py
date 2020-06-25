#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:46:46 2020

@author: aarav
"""


# Reference: https://stackoverflow.com/a/23312964/5370202
import socket
import struct
import sys
import netifaces as ni
import os
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing import image
dir_path = os.path.dirname(os.path.realpath(__file__))
print('dir path is \n ', dir_path)
sys.path.insert(0, dir_path+"/Modules")
import utils
import FaceEliminator
from tensorflow import keras

# loading the model
model = keras.models.load_model('/home/aarav/Desktop/MajorProject/Models/m2.h5')

def predictSign(img):
    hand=img
    hand= cv2.resize(hand, (64,64))
    img = image.img_to_array(hand)
    img = np.expand_dims(img, axis = 0)
    img.astype('float32')
    img=img/255.0
    pred= model.predict_classes(img)
    return utils.getSign(pred[0])


socketTimeOutEnable = False
displayWindows = True
recognitionMode="SIGN"


port = int(input("Enter port no: "))

# Reference: https://stackoverflow.com/a/24196955/5370202
print(ni.interfaces())    
ni.ifaddresses('wlp1s0')
ipAddr = ni.ifaddresses('wlp1s0')[ni.AF_INET][0]['addr']
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
print("TCP Socket successfully created")
s.bind(('', port))
print("TCP Socket binded to %s: %s" %(ipAddr,port))
s.listen(1)
print("Socket is listening")
client, addr = s.accept()     
print('Got TCP connection from', addr)


while True:

    buf = client.recv(4)

    # print(buf)
    size = struct.unpack('!i', buf)[0]  
    #Reference: https://stackoverflow.com/a/37601966/5370202, https://docs.python.org/3/library/struct.html
    # print(size)
    print("receiving image of size: %s bytes" % size)
 
    data = client.recv(size,socket.MSG_WAITALL)  #Reference: https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/
    
    # Instead of storing the image as mentioned in the 1st reference: https://stackoverflow.com/a/23312964/5370202
    # we can directly convert it to Opencv Mat format
    # Reference: https://stackoverflow.com/a/17170855/5370202
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = imutils.rotate_bound(img_np,90)
    img_np = cv2.resize(img_np,(0,0), fx=0.7, fy=0.7)
    
    mask1 = utils.segment(img_np)
    

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    haar_cascade_face = cv2.CascadeClassifier('/home/aarav/Desktop/MajorProject/Models/haarcascade_frontalface_default.xml')
    rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    maxArea1 = 0
    faceRect = -1
    foundFace = False
    for (x,y,w,h) in rects:
        if w*h > maxArea1:
            maxArea1 = w*h
            faceRect = (x,y,w,h)
            foundFace = True  
      
    mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)
    
    if displayWindows:
        cv2.imshow("Mask12",mask1)

  
    if displayWindows:
        cv2.imshow("Originl Img",img_np)


    # contour of hand is useless delete krna h isko 
    handFound, hand, contours_of_hand = utils.get_my_hand(mask1)

    if recognitionMode == "SIGN":
        if handFound:
            if displayWindows:
                cv2.imshow("Your hand",hand)            
            pred = predictSign(hand)
        else:
            pred = -1
        utils.addToQueue(pred)
        pred = utils.getConsistentSign(displayWindows)

        print("Stable Sign:",pred)

        if pred == -1:
            op1  = "--"+"\r\n"
        else:
            if pred == "2":
                pred = "2 / v"
            op1 = pred+"\r\n"
   

    if recognitionMode =="SIGN":
        client.send(op1.encode('ascii'))
    

    k = cv2.waitKey(10)
    if k == 'q':
        break

print('Stopped TCP server of port: '+str(port))
print(recognitionMode+" recognition stopped")



s.close()
cv2.destroyAllWindows()
