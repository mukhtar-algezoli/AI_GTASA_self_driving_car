import cv2
from PIL import ImageGrab
import numpy as np
import time
from directKeys import PressKey,W,A,S,D,ReleaseKey
from getKeys import key_check
import os
from AlexNetPytorch import*
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch
from IPython.core.debugger import set_trace
import visdomvisualize
import PIL


PATH = "savedmodel.tar"

transform = transforms.Compose([
    # you can add other transformations in this list
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])

def straight():
    print("STRAIGHT")
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
def right():
    print("RIGHT")
    ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)

def left():
    print("LEFT")
    ReleaseKey(W)
    ReleaseKey(D)
    PressKey(A)

    
def forward_left():
    print("FORWARD_LEFT")
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def forward_right():
    print("FORWARD_RIGHT")
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)


# def keys_to_output(keys):
    # output = [0,0,0,0,0]
    
    # if 'A' in keys:
        # output[0] = 1
    # elif 'D' in keys:
        # output[1] = 1
    # elif ('D' in keys) & ('W' in keys):
        # output[3] = 1
    # elif ('A' in keys) & ('W' in keys):
        # output[4] = 1        
    # else:
        # output[2] = 1
    # return output

# file_name = "training_data2.npy"
# if os.path.isfile(file_name):
   # print("file exists , loading previous data!")
   # training_data = list(np.load(file_name))
# else:
   # print("file does not exist , starting fresh")
   # training_data = []   
    
last_time = time.time()

AlexNet = AlexNet()

criterion = nn.MSELoss()
optimizer = optim.SGD(AlexNet.parameters(), lr=0.001, momentum=0.9)


if os.path.exists(PATH):
  checkpoint = torch.load(PATH)
  AlexNet.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # AlexNet.eval()
  print("checkpoint loaded")


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
   # screen_output = torch.from_numpy(screen_output)
   
   screen_output = PIL.Image.fromarray(screen_output)
   screen_output = transform(screen_output)
   screen_output = torch.mul(screen_output , 255)
   screen_output = torch.unsqueeze(screen_output, 1)
   print(screen_output)
   nn_output = AlexNet(screen_output)
   _, predicted = torch.max(nn_output, 1)
   print(predicted.item())
   final_output = [0,0,0,0,0]
   final_output[predicted.item()] = 1
   print(final_output)
   if final_output == [1,0,0,0,0]:
            left()
   elif final_output == [0,1,0,0,0]:
            right()              
   elif final_output == [0,0,1,0,0]:
            straight()            
   elif final_output == [0,0,0,1,0]:
            forward_right()
   elif final_output == [0,0,0,0,1]:
            forward_left()            
   
   if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break   
    
   # if len(training_data) % 500 == 0:
       # print(len(training_data))
       # np.save(file_name,training_data)   
