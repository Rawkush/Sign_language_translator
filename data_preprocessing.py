#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:16:15 2020
@author: aarav
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import utils
dir_path = os.path.dirname(os.path.realpath(__file__))
print('dir path is \n ', dir_path)
sys.path.insert(0, dir_path+"/Modules")

test_image = cv2.imread('/home/aarav/Desktop/1.png')
test_image = cv2.resize(test_image,(284,284))    


  
def get_my_hand(image_skin_mask, mask1):
    """
    ### Hand extractor
    __DO NOT INCLUDE YOUR FACE IN THE `image_skin_mask`__
    Provide an image where skin areas are represented by white and black otherwise.
    This function does the hardwork of finding your hand area in the image.
    Returns: *(image)* Your hand, *(hand_contour)*.
    """
    _,contours,_ = cv2.findContours(image_skin_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    ci = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
    if ci == -1:
        return [ False, None, None  ]
    x,y,w,h = cv2.boundingRect(contours[ci])
    # hand = np.zeros((image_skin_mask.shape[1], image_skin_mask.shape[0], 1), np.uint8)
    # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
    # _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
    hand = mask1[y:y+h,x:x+w]
    
    return [ True, hand, contours[ci] ]


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

              
            
#read_from_folder('Digits/')
#read_from_folder('Letters/')
mask1 = segment(test_image)
'''
src1_mask=cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
out=cv2.subtract(src1_mask,test_image)
out=cv2.subtract(src1_mask,out)
'''
handFound, hand, contours_of_hand = utils.get_my_hand(test_image, mask1)
