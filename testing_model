#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:52:24 2020

@author: aarav
"""


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
from keras.preprocessing import image
# Following modules are used specifically for Gesture recognition

currentModuleName = __file__.split(os.path.sep)[-1]
print('current modelu name \n',currentModuleName)


from tensorflow import keras
model = keras.models.load_model('/home/aarav/Desktop/MajorProject/Models/my_model.h5')

test_image = cv2.imread('/home/aarav/Desktop/MajorProject/Dataset/Digits/7/1.png')



gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
handFound, hand, contours_of_hand = utils.get_my_hand(gray)
hand= cv2.resize(hand, (64,64))
hand=hand/255
img = image.img_to_array(hand)
img = np.expand_dims(img, axis = 0)
pred= model.predict_classes(img)