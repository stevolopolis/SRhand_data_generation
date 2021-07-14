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
import mediapipe as mp
import cv2 as cv 
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

VISUALIZE = True
keypoints = [0, 1, 5, 9, 13, 17]

device = torch.device('cuda')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#target = sys.argv[1] if len(sys.argv) > 1 else None
target = 0
model = torch.jit.load('hand.pts', map_location=device)
print('target... ', target)
print('loading model... done')

save_annotation_path = "./annotations"
vis_path = "./vis"
TRAIN_IMAGE_HEIGHT = TRAIN_IMAGE_WIDTH = 256
LABEL_MIN = 0.3
LABEL_HAND_MIN = 0.2
FOLD_RATIO_THRESHOLD = [100]*5

PORT = 9999

HAND_COLORS = [
    (100, 100, 100), (100, 0, 0), (150, 0, 0), (200, 0, 0), (255, 0, 0), (100, 100, 0), (150, 150, 0),
    (200, 200, 0), (255, 255, 0), (0, 100, 50), (0, 150, 75), (0, 200, 100), (0, 255, 125),
    (0, 50, 100), (0, 75, 150),(0, 100, 200),(0, 125, 255),(100, 0, 100),(150, 0, 150),(200, 0, 200), (255, 0, 255)
]
_min = _max = None

def write_annotation(img,pts,hand_rect):
    
    #get time for naming the images
    #current_time = datetime.datetime.now().timestamp() * 1000
    #txt_file = current_time + ".txt"
    #img_file = current_time+".jpg"

    assert (len(pts) == len(hand_rect)) #make sure they have the same len before writing the annotation
    
    for pt in pts:
        #input(pt) #convert tuple to list first

        tmp = [x for x in pt if x] #handle empty list in keypoints
        
        tmp_ary = np.array(tmp)
        pt = np.array(pt)
        
        min_x = min(tmp_ary[:,0]) #@ERROR, Empty list()
        min_y = min(tmp_ary[:,1])

        max_x = max(tmp_ary[:,0])
        max_y = max(tmp_ary[:,1])
        
        w = max_x - min_x #original ratio
        h = max_y - min_y

        #center points
        (cen_x,cen_y) = (int(min_x+(w/2)),int(min_y+(h/2)))
        
        
        if w <= h: #make it square
            box_size = h
        else:
            box_size = w

        
        box_size = math.ceil(box_size*1.4) #pad with ratio

        (new_min_x,new_min_y) = (int(cen_x - box_size/2),int(cen_y - box_size/2))
        (new_max_x,new_max_y) = (int(cen_x + box_size/2),int(cen_y + box_size/2))
            
        #calculate all the new points relative to the bounding box_size
        #all the points (x - min_x, y - min_y)
        
        kp_array = []
        
        #replace all empty lost to zeros
        #print(pt)
        #print(type(pt))
        for idx,p in enumerate(pt):
            #print(p)
            #print(type(p))
            if len(p) is 0:
                pt[idx] = tuple([0,0])


        
        #print(pt.dtype)
        #relative_pts_coord = np.array([]).astype("int")
        relative_pts_coord = []
        for p in pt:
            #print(p)
            #relative_pts_coord = np.append(relative_pts_coord,p)
            relative_pts_coord.append(p)
        #relative_pts_coord = pt.copy() #for checking
        
        #relative_pts_coord = zip(relative_pts_coord,relative_pts_coord[1:])[::2]
        pt = copy.deepcopy(relative_pts_coord)
        #for p in range(pt.shape[0]): #go through 21 points
        for p, item in enumerate(pt):   
            #pt[p] = np.array(pt[p])
            #relative_pts_coord[p] = np.array(relative_pts_coord[p])
            if len(pt[p]) == 0: #empty list return false
                continue
            #if pt[p,1]!=0:
            if len(pt[p]) is not 0:
                #relative_pts_coord[p,0] = int(pt[p,0] - new_min_x)
                #relative_pts_coord[p,1] = int(pt[p,1] - new_min_y)
                try:
                    #print(pt[0])
                    #print(pt[0][0])
                    #print(relative_pts_coord[0])
                    #print(relative_pts_coord[0][0])
                    #print(relative_pts_coord[p][0])
                    #print(relative_pts_coord[p][1])
                    #print(pt[p][0])
                    #print(pt[p][1])
                    #print(new_min_x)
                    #print(new_min_y)
                    #relative_pts_coord[p][0] = int(pt[p][0] - new_min_x)
                    #relative_pts_coord[p][1] = int(pt[p][1] - new_min_y)
                    if pt[p][0] == 0 or pt[p][1] == 0:
                        relative_pts_coord[p] = (int(0),int(0))
                        kp_array.append(0)
                        kp_array.append(0)
                    else:
                        relative_pts_coord[p] = (int(pt[p][0] - new_min_x),int(pt[p][1] - new_min_y))
                        kp_array.append(int(pt[p][0] - new_min_x))
                        kp_array.append(int(pt[p][1] - new_min_y))
                except:
                    input("error")
                    np.append(relative_pts_coord[p],0)
                    np.append(relative_pts_coord[p],0)
                    kp_array.append(int(0))
                    kp_array.append(int(0))
                    continue
        
        pt = copy.deepcopy(relative_pts_coord)
                
        kp_array = np.asarray(kp_array)
        
        target_size = 192
        hand_img = img[new_min_y:new_max_y,new_min_x:new_max_x]
        if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
            return False
        hand_img = cv2.resize(hand_img,(target_size,target_size))
        hand_img_vis = hand_img.copy()
        scale = target_size/box_size
        kp_array = kp_array*scale
    
        #for p in range(pt.shape[0]): #go through 21 points
        for p, item in enumerate(pt):
            if len(pt[p]) is not 0: #valid point
                if relative_pts_coord[p][0] is 0 or relative_pts_coord[p][1] is 0: #dont draw
                    continue
                else:
                    hand_img_vis = cv2.circle(hand_img_vis, (int(relative_pts_coord[p][0]*scale),int(relative_pts_coord[p][1]*scale)), radius=4, color=(0, 0, 255), thickness=-1)

        current_time = str(datetime.now().timestamp() * 1000)
        txt_file = current_time + ".txt"
        img_file = current_time+".jpg"
        
        #cv2.imshow("",hand_img)


        save_img_path = os.path.join(save_annotation_path,img_file)
        save_txt_path = os.path.join(save_annotation_path,txt_file)
        save_vis_path = os.path.join(vis_path,img_file)

        with open(save_txt_path,'w') as f:
            for num in kp_array:
                f.write(str(int(num))+ " ")
            cv2.imwrite(save_img_path,hand_img)
            cv2.imwrite(save_vis_path,hand_img_vis)
        f.close()
        #input("image saved")

        return True
    

def dis(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
def norm(a):
    return dis(a, (0,0))
def vector(a, b):
    v = (b[0]-a[0], b[1]-a[1])
    return v

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    if v1 == (0, 0) or v2 == (0,0):
        return 180
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ans = abs(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))
    print(v1, v2, '=', ans)
    return ans
def nmslocation(src, threshold):
    locations = []
    blockwidth = 2
    rows, cols = src.shape

    arr = feature.peak_local_max(src, min_distance=2, threshold_abs=threshold, exclude_border=True, indices=True)
    new_arr = [(src[x][y], (x, y)) for x, y in arr]
    new_arr = sorted(new_arr, key=itemgetter(0), reverse=True)
    return new_arr
    
def transform_net_input(tensor, source_img, hand_rect=None, tensor_idx=0):
    #             hand_rect.append((l_t[1], l_t[0], r_b[1], r_b[0], pos_x, pos_y))
    img = source_img.copy()
    if hand_rect is not None:
        l, t, r, b, _, _ = hand_rect[tensor_idx]
        img = img[t:b,l:r]
    rows, cols = len(img), len(img[0])
    ratio = min(tensor.shape[2] / rows, tensor.shape[3] / cols)
    mat = np.array([[ratio, 0, 0], [0, ratio, 0]])

    dst = cv2.warpAffine(img, mat, (tensor.shape[3], tensor.shape[2]))

    dst = dst / 255 - 0.5
    r, g, b = cv2.split(dst)

    tensor[tensor_idx][0] = torch.tensor(r, device=device).float()
    tensor[tensor_idx][1] = torch.tensor(g, device=device).float()
    tensor[tensor_idx][2] = torch.tensor(b, device=device).float()
    return ratio

def detect_bbox(input_image):
    tensor = torch.zeros([1, 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH], device=device)
    rows, cols, _ = input_image.shape
    ratio_input_to_net = transform_net_input(tensor, input_image)
    heatmap = model.forward(tensor)[3]
    ratio_net_downsample = TRAIN_IMAGE_HEIGHT / heatmap.shape[2]
    rect_map_idx = heatmap.shape[1] - 3

    rectmap = []
    # copy three channel rect map
    for i in range(3):
        rectmap.append(np.copy(heatmap[0][i+rect_map_idx].cpu().detach().numpy()))
    canv = np.copy(rectmap[0])
    locations = nmslocation(rectmap[0], LABEL_MIN)
    hand_rect = []
    for loc_val, points in locations:
        pos_x, pos_y = points
        ratio_width = ratio_height = pixelcount = 0
        for m in range(max(pos_x-2, 0), min(pos_x+3, int(heatmap.shape[2]))):
            for n in range(max(pos_y-2, 0), min(pos_y+3, int(heatmap.shape[3]))):
                ratio_width += rectmap[1][m][n]
                ratio_height += rectmap[2][m][n]
                pixelcount += 1

        if pixelcount > 0:
            ratio_width = min(max(ratio_width / pixelcount, 0), 1)
            ratio_height = min(max(ratio_height / pixelcount, 0), 1)
            ratio = ratio_net_downsample / ratio_input_to_net
            pos_x *= ratio
            pos_y *= ratio
            rect_w = ratio_width * TRAIN_IMAGE_WIDTH / ratio_input_to_net
            rect_h = ratio_height * TRAIN_IMAGE_HEIGHT / ratio_input_to_net

            l_t = (max(int(pos_x - rect_h/2), 0), max(int(pos_y - rect_w/2), 0))
            r_b = (min(int(pos_x + rect_h/2), rows - 1), min(int(pos_y + rect_w/2), cols - 1))

            hand_rect.append((l_t[1], l_t[0], r_b[1], r_b[0], pos_x, pos_y))

    return hand_rect

def detect_hand(input_image, hand_rect):
    many_points = [None]*len(hand_rect)

    tensor = torch.zeros([len(hand_rect), 3, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH], device=device)
    ratio_input_to_net = [None]*len(hand_rect)
    for i in range(len(hand_rect)):
        ratio_input_to_net[i] = transform_net_input(tensor, input_image, hand_rect, i)

    net_result = model.forward(tensor)[3]
    ratio_net_downsample = TRAIN_IMAGE_HEIGHT / net_result.size()[2]
    heatmaps = []*len(hand_rect)
    many_points = []
    for rect_idx in range(len(hand_rect)):
        total_points = [[] for i in range(21)]
        x, y, _, _, _, _ = hand_rect[rect_idx]
        ratio = ratio_net_downsample / ratio_input_to_net[rect_idx]
        for i in range(21):
            heatmap = net_result[rect_idx][i].cpu().detach().numpy()
            points = nmslocation(heatmap, LABEL_HAND_MIN)
            if len(points):
                _, point = points[0]
                total_points[i] = (int(point[1]*ratio)+x, int(point[0]*ratio)+y)
        many_points.append(total_points)
    return many_points

def detect_hand_v2(input_image):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        many_points = []
        kp_list = [(0,0) for i in range(21)]
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for keypoint in range(21):
                coord = (int(hand_landmarks.landmark[keypoint].x * image.shape[1]), int(hand_landmarks.landmark[keypoint].y * image.shape[0]))
                kp_list[keypoint] = coord

        many_points.append(kp_list)

    return many_points

def pyramid_inference(input_image):

    rows, cols, _ = input_image.shape
    hand_rects = detect_bbox(input_image)
    if len(hand_rects) == 0:
        return [], []

    #many_keypoints = detect_hand(input_image, hand_rects)
    many_keypoints = detect_hand_v2(input_image)
    for i in range(len(hand_rects)-1, -1, -1):
        missing_points = 0
        for j in range(21):
            if len(many_keypoints[i][j]) != 2:
                missing_points += 1
        if missing_points > 5:
            hand_rects.pop(i)
            many_keypoints.pop(i)

    return many_keypoints, hand_rects

def feed_frame(frame):
    save_img = frame.copy()
    many_keypoints, hand_rect = pyramid_inference(frame)

    #extract kpts and hand rectangle
    have_hand = write_annotation(save_img,many_keypoints,hand_rect)

    if have_hand:
        for rect_idx, points in enumerate(many_keypoints):
            rect = hand_rect[rect_idx]
            frame = cv2.rectangle(frame, rect[0:2], rect[2:4], (0, 0, 255), 6)
            missing = 0
            if rect is None:
                continue
            for i, point in enumerate(points):
                if point is None or len(point) == 0:
                    missing+=1
                    continue
                frame = cv2.circle(frame, point, 6, HAND_COLORS[i], 6)

            per = 'bruh'
            for i in range(5):
                for j in range(3):
                    cnt = j+i*4+1
                    if len(points[cnt]) != 0 and len(points[cnt+1])!=0 :
                        frame = cv2.line(frame, points[cnt], points[cnt+1], (0, 255, 0), 2)

            text_pos = hand_rect[rect_idx][0:2]
            text_pos = (text_pos[0], text_pos[1]+5)
            frame = cv2.putText(frame, per, text_pos, 1, 3, (0, 0, 255), 3)
        frame = cv2.resize(frame, (512,512))
        #if target:
        #cv2.imshow('show', frame)
        #cv2.waitKey(0)
        return frame
    else:
        return cv2.resize(frame, (512, 512))

def main():
    cap = cv2.VideoCapture(target)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        _, frame = cap.read()
        #print(frame)
        #cv2.imshow("test", frame)
        #input()
        if frame is None:
            break
        frame = cv.flip(frame, 1)
        tmp = feed_frame(frame)
        tmp = cv2.resize(tmp,(640,480))
        cv2.imshow("show",tmp)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()

