import cv2
import numpy as np
from math import sqrt
import torch
import sys
import os
from operator import itemgetter
from skimage import feature
from datetime import datetime
import math
import copy
import mediapipe as mp
import cv2 as cv 
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

data_path = 'other_data/test'
save_annotation_path = "other_data/annotations_test"
vis_path = "other_data/vis_test"

processed_img = 0
for file in os.listdir(data_path):
    if file[-4:] != '.jpg':
        continue

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv.imread(os.path.join(data_path, file))
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        kp_list = [(0,0) for i in range(21)]
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for keypoint in range(21):
                coord = (int(hand_landmarks.landmark[keypoint].x * image.shape[1]), int(hand_landmarks.landmark[keypoint].y * image.shape[0]))
                kp_list[keypoint] = coord

    kp_array = np.asarray(kp_list)
        
    hand_img_vis = image.copy()
    scale = 1

    #for p in range(pt.shape[0]): #go through 21 points
    for p, item in enumerate(kp_array):
        if len(kp_array[p]) != 0: #valid point
            if kp_array[p][0] == 0 or kp_array[p][1] == 0: #dont draw
                continue
            else:
                hand_img_vis = cv.circle(hand_img_vis, (int(kp_array[p][0]*scale),int(kp_array[p][1]*scale)), radius=4, color=(0, 0, 255), thickness=-1)

    for i in range(5):
        for j in range(3):
            cnt = j+i*4+1
            if len(kp_array[cnt]) != 0 and len(kp_array[cnt+1])!=0 :
                frame = cv.line(hand_img_vis,
                                    (int(kp_array[cnt][0]*scale),int(kp_array[cnt][1]*scale)),
                                    (int(kp_array[cnt+1][0]*scale),int(kp_array[cnt+1][1]*scale)),
                                    (0, 255, 0),
                                    2)


    current_time = str(datetime.now().timestamp() * 1000)
    txt_file = current_time + ".txt"
    img_file = current_time+".jpg"
    
    save_img_path = os.path.join(save_annotation_path,img_file)
    save_txt_path = os.path.join(save_annotation_path,txt_file)
    save_vis_path = os.path.join(vis_path,img_file)

    with open(save_txt_path,'w') as f:
        for coords in kp_array:
            for coord in coords:
                f.write(str(int(coord))+ " ")
        cv.imwrite(save_img_path,image)
        cv.imwrite(save_vis_path,hand_img_vis)
    f.close()

    processed_img += 1

    if processed_img % 100 == 0:
        print('Processed %s images...' % processed_img)
