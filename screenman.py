import cv2
from PIL import ImageGrab
import numpy as np


while True:
   get_screen = ImageGrab.grab(bbox=(10,10,1280,720))
   screen_shot = np.array(get_screen)
   hsv = cv2.cvtColor(screen_shot , cv2.COLOR_BGR2HSV)
   lower_color = np.array([0 , 0 , 0])
   upper_color = np.array([100 , 100 , 100])
   
   kernel = np.ones((5,5),np.uint8)

   
   mask = cv2.inRange(hsv , lower_color , upper_color)
   
   output = cv2.Canny(mask , threshold1 = 50 , threshold2 = 300)
   

   
   resized = cv2.resize(output , (640 , 480))
   cv2.imshow('manipulated' , resized)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cv2.destroyAllWindows()
  