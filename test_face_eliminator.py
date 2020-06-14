#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing face eliminator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, dir_path+"/Modules")
import FaceEliminator
import utils


#Loading the image to be tested
test_image = cv2.imread('/home/aarav/Desktop/1.png')


def segment(src_img):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.

    Returns: mask
    """
    
    import cv2
    from numpy import array, uint8

    blurred_img = cv2.GaussianBlur(src_img,(5,5),0)
    blurred_img = cv2.medianBlur(blurred_img,5) 
    blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2YCrCb)
    lower = array([0,137,100], uint8)
    upper = array([255,200,150], uint8)
    mask = cv2.inRange(blurred_img, lower, upper)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def a():
    haar_cascade_face = cv2.CascadeClassifier('/home/aarav/Desktop/MajorProject/Models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    mask1 = segment(test_image)
    plt.imshow(mask1)
    rects = haar_cascade_face.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    maxArea1 = 0
    faceRect = -1
    foundFace = False
    for (x,y,w,h) in rects:
        if w*h > maxArea1:
            maxArea1 = w*h
            faceRect = (x,y,w,h)
            foundFace = True  
            
  
    for (x,y,w,h) in rects:

        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(convertToRGB(test_image))
    mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)
    plt.imshow(mask1)


test_image = cv2.resize(test_image,(284,284))    
mask1 = segment(test_image)
plt.imshow(mask1, cmap='gray')
handFound, hand, contours_of_hand = utils.get_my_hand(mask1)
plt.imshow(hand, cmap='gray')
