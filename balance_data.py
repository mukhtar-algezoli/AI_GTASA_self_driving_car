import numpy as np
import pandas as pd
from random import shuffle
import cv2

train_data = np.load('training_data3.npy')

shuffle(train_data)

lefts = []
rights = []
forwards = []
forwards_rights = []
forwards_lefts = []

for data in train_data:
  img = data[0]
  choice = data[1]
  
  if choice == [1,0,0,0,0]:
    lefts.append([img,choice])
  elif choice == [0,1,0,0,0]:
   rights.append([img,choice])
  elif choice == [0,0,1,0,0]:
    forwards.append([img,choice])
  elif choice == [0,0,0,1,0]:
    forwards_rights.append([img,choice])
  elif choice == [0,0,0,0,1]:
    forwards_lefts.append([img,choice])    
  else:
    print('no matches')  
    
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]        
forwards_rights = forwards_rights[:len(forwards)]        
forwards_lefts = forwards_lefts[:len(forwards)]       

final_data = forwards + lefts + rights
shuffle(final_data)

np.save('training_data_v2.npy', final_data) 

# for data in train_data:
  # img = data[0]
  # choise = data[1]
  # cv2.imshow('test' , img)
  # print(choise)
  # if cv2.waitKey(25) & 0xFF == ord('q'):
   # cv2.destroyAllWindows()
   # break