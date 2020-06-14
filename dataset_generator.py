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
import utils

test_image = cv2.imread('/home/aarav/Desktop/1.png')
test_image = cv2.resize(test_image,(284,284))    


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

def getHandImage(scr_img):
    handFound, hand, contours_of_hand = utils.get_my_hand(mask1)
    return hand
    

def read_from_folder(dir_):
    """
    Loads data and preprocess and save the images
    """
    images = []
    labels = []
    img_dir='/home/aarav/Downloads/Compressed/' + dir_
    
    dataset_dir='/home/aarav/Desktop/MajorProject/Dataset/' +dir_
    
    print("LOADING DATA FROM Digits: ",end = "")
    
    
    for folder in os.listdir(img_dir):
        print(folder, end = ' | ')
        
        #making sub folder fo alphabets
        if not os.path.exists(dataset_dir + folder):
            os.makedirs(dataset_dir + folder)
        
        for image in os.listdir(img_dir + "/" + folder):
            print(image)
            if image.endswith('txt'):
                continue
            temp_img = cv2.imread(img_dir + '/' + folder + '/' + image)
            mask1 = segment(temp_img)
            handFound, hand, contours_of_hand = utils.get_my_hand(mask1)
            if(handFound):
                cv2.imwrite( dataset_dir + folder + '/' +image ,hand)
            
            
            
            
read_from_folder('Digits/')
read_from_folder('Letters/')
