import cv2
from PIL import ImageGrab
import numpy as np


while True:
   get_screen = ImageGrab.grab(bbox=(10,10,1280,720))
   screen_shot = np.array(get_screen)
   # screen_shot = cv2.cvtColor(screen_shot , cv2.COLOR_BGR2HSV)
   
   screen_shot = cv2.cvtColor(screen_shot , cv2.COLOR_BGR2GRAY)
   # ret2 , screen_shot = cv2.threshold(screen_shot , 50 , 255 , cv2.THRESH_BINARY)
   # screen_shot = cv2.adaptiveThreshold(screen_shot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
   screen_shot = cv2.Canny(screen_shot , threshold1 = 50 , threshold2 = 300)
   
   resized = cv2.resize(screen_shot , (640 , 480))
   cv2.imshow('screen_shot' , resized)
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cv2.destroyAllWindows()
  