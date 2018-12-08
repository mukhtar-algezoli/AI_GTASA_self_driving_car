import cv2
from PIL import ImageGrab
import numpy as np

img = cv2.imread('black.jpg')
for i in range(127):
  for j in range(72):
    cv2.drawMarker(img, (i*10 , j*10), (255,0,255),markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1, line_type=8)    

while True:

   kernel = np.ones((15 , 15) , np.float32)/225
   get_screen = ImageGrab.grab(bbox=(0,0,1280,720))
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
   screen_with_marker = cv2.drawMarker(screen_shot, (200,200), (0,0,255),markerType=cv2.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=8)

   
   
   resized = cv2.resize(output , (640 , 480))
   the_screen = cv2.resize(screen_with_marker , (640 , 480))
   cv2.imshow('manipulated' , resized)
   cv2.imshow('with_marker' , img)
   
   
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cv2.destroyAllWindows()
  