
# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import atexit
import struct
import sys
import netifaces as ni
import os
import numpy as np
import cv2
import imutils
from imutils import face_utils
import pickle
dir_path = os.path.dirname(os.path.realpath(__file__))
print('dir path is \n ', dir_path)
import utils
sys.path.insert(0, dir_path+"/Modules")
import FaceEliminator

# Following modules are used specifically for Gesture recognition

currentModuleName = __file__.split(os.path.sep)[-1]
print('current modelu name \n',currentModuleName)

gridSize = (10,10)

recognitionMode = "SIGN"  # SIGN    # This is the mode of recognition. 



socketTimeOutEnable = False

noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket

total_captured=601  # This is used as an initial count of frames captured for capturing new frames


lastMsgSentOut = '--\r\n'


####detector = dlib.get_frontal_face_detector()

videoCounter = 1

recordVideos = False

displayWindows = True

if recognitionMode == "SIGN":
    print("Check 1")
    classifier = pickle.load(open('/home/aarav/Desktop/MajorProject/Models/my_model.sav','rb'))
    print("Check 2")
    print("Loaded Sign Recognition KNN Model")

else:
    print("video mode not developed yet")

def port_initializer():
    global port
    port = int(port_entry.get())
    opening_window.destroy()


port = int(input("Enter port no: "))

# Reference: https://stackoverflow.com/a/24196955/5370202
print(ni.interfaces())    
ni.ifaddresses('wlp1s0')
ipAddr = ni.ifaddresses('wlp1s0')[ni.AF_INET][0]['addr']
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if socketTimeOutEnable:
    s.settimeout(20)   
print("TCP Socket successfully created")
s.bind(('', port))
print("TCP Socket binded to %s: %s" %(ipAddr,port))
s.listen(1)
print("Socket is listening")
client, addr = s.accept()     
print('Got TCP connection from', addr)
if socketTimeOutEnable:
    s.settimeout(10)


while True:
    
    noOfFramesCollected += 1
    if displayWindows:
        utils.displayTextOnWindow("Frame No",str(noOfFramesCollected))
    
    
    buf = client.recv(4)

    # print(buf)
    size = struct.unpack('!i', buf)[0]  
    #Reference: https://stackoverflow.com/a/37601966/5370202, https://docs.python.org/3/library/struct.html
    # print(size)
    print("receiving image of size: %s bytes" % size)

    if(size == 0 and recognitionMode == "SIGN"):
        op1 = "QUIT\r\n"
        client.send(op1.encode('ascii'))
        break
   
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

    '''
    #rects = detector(gray, 1)
    maxArea1 = 0
    faceRect = -1
    foundFace = False
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if w*h > maxArea1:
            maxArea1 = w*h
            faceRect = (x,y,w,h)
            foundFace = True
           
    '''        
    mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)
    if displayWindows:
        cv2.imshow("Mask12",mask1)

  
    if displayWindows:
        cv2.imshow("Originl Img",img_np)


# contour of hand is useless delete k rna h isko 
    handFound, hand, contours_of_hand = utils.get_my_hand(mask1)

    if recognitionMode == "SIGN":
        if handFound:
            if displayWindows:
                cv2.imshow("Your hand",hand)
                
            features = utils.extract_features(hand, gridSize)
            pred = utils.predictSign(classifier,features)
        else:
            pred = -1
        utils.addToQueue(pred)
        pred = utils.getConsistentSign(displayWindows)

        # pred = -1
        print("Stable Sign:",pred)

        if pred == -1:
            op1  = "--"+"\r\n"
        else:
            if pred == "2":
                pred = "2 / v"
            op1 = pred+"\r\n"

    else:
        break
    
    

    if recognitionMode =="SIGN":
        client.send(op1.encode('ascii'))
        lastMsgSentOut = op1
    

    k = cv2.waitKey(10)
    if k == 'q':
        break
    
    


print('Stopped TCP server of port: '+str(port))
print(recognitionMode+" recognition stopped")



s.close()
cv2.destroyAllWindows()




def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)