#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:18:36 2020

@author: aarav
"""

import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image


test_image = cv2.imread('/home/aarav/Desktop/1.png')
test_image = cv2.resize(test_image,(256,256))    
img = image.img_to_array(test_image)
img = np.expand_dims(img, axis = 0)
inp = tf.convert_to_tensor(img, np.float32)

interpreter = tf.lite.Interpreter(model_path="/home/aarav/Desktop/hand_landmark.tflite")
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_details[0]['shape']

print(output_details[0]['shape'])  # Example: [1 224 224 3]
print(output_details[0]['dtype'])  # Example: <class 'numpy.float32'>


interpreter.set_tensor(input_details[0]['index'], inp)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)








nPoints = 21

i=0
x=[]
y=[]
z=[]
for i in output_data[0]:
    if i<21:
        x.append(i)
    elif i<42:
        y.append(i)
    else:
        z.append(i)


test_image[x[0],y[0]] = [0,0,0]

plt.imshow(img)









