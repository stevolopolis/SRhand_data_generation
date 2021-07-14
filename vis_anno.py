#!/usr/bin/python3
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
import glob


#target = sys.argv[1] if len(sys.argv) > 1 else None
root_folder = "/home/leochoi/Downloads/SRhand/inspected_anno"
mode_folder = "train"

save_folder = "/home/leochoi/Downloads/SRhand/vis_annotation"
#TRAIN_IMAGE_HEIGHT = TRAIN_IMAGE_WIDTH = 256
#LABEL_MIN = 0.3
#LABEL_HAND_MIN = 0.2
#FOLD_RATIO_THRESHOLD = [100]*5
path = os.path.join(root_folder,mode_folder)
image_paths = os.path.join(path,"*.jpg")
txt_paths = os.path.join(path,"*.txt")

img_files = glob.glob(image_paths)
txt_files = glob.glob(txt_paths)

assert(len(img_files) == len(txt_files))

img_files = sorted(img_files)
txt_files = sorted(txt_files)


for idx, img in enumerate(img_files):
    save_vis_path = os.path.join(save_folder,os.path.basename(img))
    hand_img = cv2.imread(img)
    hand_img_vis = hand_img.copy()

    print(txt_files[idx])
    with open(txt_files[idx], 'r') as f:
        pt = f.readline()
        pt = np.fromstring(pt, dtype=int, sep=' ').tolist()
        pt = [*zip(pt[::2], pt[1::2])]
        #print(pt)

        #input(type(pt))
    f.close()


    for p, item in enumerate(pt):
        if len(pt[p]) is not 0: #valid point
            if pt[p][0] is 0 or pt[p][1] is 0: #dont draw
                continue
            else:
                hand_img_vis = cv2.circle(hand_img_vis, (int(pt[p][0]),int(pt[p][1])), radius=4, color=(0, 0, 255), thickness=-1)
            
    cv2.imwrite(save_vis_path,hand_img_vis)
