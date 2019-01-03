import cv2
from PIL import ImageGrab
import numpy as np
import time
from directKeys import PressKey,W,A,S,D
from getKeys import key_check
import os


def keys_to_output(keys):
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

file_name = "training_data.npy"
if os.path.isfile(file_name):
   print("file exists , loading previous data!")
   training_data = list(np.load(file_name))
else:
   print("file does not exist , starting fresh")
   training_data = []   
    
last_time = time.time()


while True:


   kernel = np.ones((15 , 15) , np.float32)/225
   get_screen = ImageGrab.grab(bbox=(10,10,1280,720))
   screen_shot = np.array(get_screen)
   hsv = cv2.cvtColor(screen_shot , cv2.COLOR_BGR2HSV)
   lower_color = np.array([90 , 0 , 70])
   upper_color = np.array([100 , 100 ,  100])
   
   
   output = cv2.inRange(hsv , lower_color , upper_color)
   
   kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
   dilation = cv2.dilate(output, kernel, iterations=1)
   output = cv2.erode(dilation, kernel, iterations=1)   
   # output = cv2.Canny(output , threshold1 = 50 , threshold2 = 300)
   # output = cv2.GaussianBlur(output , (15,15) , 0)

   
   resized = cv2.resize(output , (640 , 480))
   
   print('loop took {} seconds'.format(time.time()-last_time))
   last_time = time.time()
   
   
   cv2.imshow('manipulated' , resized)
   screen_output = cv2.resize(output , (224 ,224))
   keys = key_check()
   Keys_output = keys_to_output(keys)
   training_data.append([screen_output,Keys_output])   
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break   
    
   if len(training_data) % 500 == 0:
       print(len(training_data))
       np.save(file_name,training_data)   
