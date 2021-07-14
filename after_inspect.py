import os

import random

save_path = "./inspected_anno"

inspected_img_list = os.listdir("./inspected")
data_path = "./annotations"
random.shuffle(inspected_img_list)

#80% training data
train_amount = int(0.8*len(inspected_img_list))

for idx, img in enumerate(inspected_img_list):

    if idx < train_amount:
        mode = "train"
    else:
        mode = "test"
    
    save_path_tmp = os.path.join(save_path,mode)

    txt_path = os.path.join(data_path,img[:-3]+"txt")
    img = os.path.join(data_path,img)
    os.system("cp " + txt_path + " " + save_path_tmp)
    os.system("cp " + img + " " + save_path_tmp)
    #input(0)