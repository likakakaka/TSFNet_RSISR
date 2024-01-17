import glob
import os
import random
import shutil
import torch
import random
datapath='G:\datasets\yaogan\dataset/NWPU-RESISC45/NWPU-RESISC45'
valpath='G:\datasets\yaogan\dataset/NWPU-RESISC45/val/HR'
trainpath='G:\datasets\yaogan\dataset/NWPU-RESISC45/train/HR'
testpath='G:\datasets\yaogan\dataset/NWPU-RESISC45/test/HR'
img_list = os.listdir(datapath)

for dir in img_list:
    list_class_img= os.listdir(datapath+'/'+dir)
    random.shuffle(list_class_img)
    for ind in range(len(list_class_img)):
        img_name=list_class_img[ind]
        file_name = os.path.join(datapath,dir, img_name)
        if ind<45:
            dir_train = os.path.join(trainpath, img_name)
            shutil.copy(file_name, dir_train)
        elif ind<50:
            dir_val = os.path.join(valpath, img_name)
            shutil.copy(file_name, dir_val)
        elif ind<100:
            dir_test = os.path.join(testpath, img_name)
            shutil.copy(file_name, dir_test)
        else:
            break
