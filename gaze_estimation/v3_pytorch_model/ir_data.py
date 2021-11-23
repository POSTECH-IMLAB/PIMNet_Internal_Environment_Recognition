import sys
import torch.utils.data as data
from os import listdir
import os
import random
import torch
import cv2
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def is_usable_gaze(filename):
    GAZE_ZONES = ['part_1', 'part_3', 'part_6', 'part_8', 'part_10', 'part_12']
    filename_lower = filename.lower().split('.')[0]
    return any(filename_lower.endswith(gaze_zone) for gaze_zone in GAZE_ZONES)

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            return img
    except IOError:
        print('Cannot load image ' + path)

class IR_FACE_Dataset(data.Dataset):
    def __init__(self, data_path, img_w, img_h, img_local_h, transform, loader=img_loader,\
         with_subfolder=False, random_crop=True, read_fld=True, return_name=False):
        super(IR_FACE_Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]



        # <pick specific zones>
        self.samples = [x for x in self.samples if is_usable_gaze(x)]
        '''
        data_list_1 = glob('F:/DB/MOBIS/CROPPED_2/*part_1.jpg') #1
        data_list_2 = glob('F:/DB/MOBIS/CROPPED_2/*part_3.jpg') #2
        data_list_3 = glob('F:/DB/MOBIS/CROPPED_2/*part_6.jpg')  #3
        data_list_4 = glob('F:/DB/MOBIS/CROPPED_2/*part_8.jpg')  #4
        data_list_5 = glob('F:/DB/MOBIS/CROPPED_2/*part_10.jpg')  #5
        data_list_6 = glob('F:/DB/MOBIS/CROPPED_2/*part_12.jpg')  #6
        '''

        self.data_path = data_path
        self.img_w = img_w
        self.img_h = img_h
        self.img_local_h = img_local_h
        self.transform = transform
        self.random_crop = random_crop
        self.return_name = return_name
        self.loader = loader

        # if true, read facial landmarks
        self.read_fld = read_fld


        print(str(len(self.samples)) + "  items found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        #path = os.path.join(self.data_path, self.samples[index])

        path = self.data_path + '/' + self.samples[index]
        
        img = self.loader(path)
        w, h = img.shape[0], img.shape[1]

        # use fld?
        if self.read_fld:
            fld_file = path.replace("jpg", "txt")
            fld_fdes = open(fld_file, "r")
            flds = np.array(fld_fdes.read().split(), dtype=np.float32)
            flds = flds.reshape(68, 2)
            fld_fdes.close()

        # need resize?
        if w < self.img_w or h < self.img_h or w > self.img_w or h > self.img_h:
            
            if self.read_fld:
                w_ratio, h_ratio = self.img_w / w, self.img_h / h 
                flds[:, 0] = flds[:, 0] * w_ratio
                flds[:, 1] = flds[:, 1] * h_ratio
            
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)


        local_img = img[0:self.img_local_h, 0:self.img_w]
        

            

        # pick class
        gaze_part = int(path.split('_')[-1].split('.')[0])
        label_tensor = np.zeros([6])

        '''
        data_list_1 = glob('F:/DB/MOBIS/CROPPED_2/*part_1.jpg') #1
        data_list_2 = glob('F:/DB/MOBIS/CROPPED_2/*part_3.jpg') #2
        data_list_3 = glob('F:/DB/MOBIS/CROPPED_2/*part_6.jpg')  #3
        data_list_4 = glob('F:/DB/MOBIS/CROPPED_2/*part_8.jpg')  #4
        data_list_5 = glob('F:/DB/MOBIS/CROPPED_2/*part_10.jpg')  #5
        data_list_6 = glob('F:/DB/MOBIS/CROPPED_2/*part_12.jpg')  #6
        '''
        if gaze_part == 1:
            gaze_class = 0
            label_tensor[0] = 1
        elif gaze_part == 3:
            gaze_class = 1
            label_tensor[1] = 1
        elif gaze_part == 6:
            gaze_class = 2
            label_tensor[2] = 1
        elif gaze_part == 8:
            gaze_class = 3
            label_tensor[3] = 1
        elif gaze_part == 10:
            gaze_class = 4
            label_tensor[4] = 1
        elif gaze_part == 12:
            gaze_class = 5
            label_tensor[5] = 1
        
        label_tensor = torch.LongTensor(label_tensor)
        #print(path + " --- " + gaze_class)


        if self.transform is not None:
            img = self.transform(img)
            local_img = self.transform(local_img)
        else:
            img = torch.from_numpy(img)
            local_img = torch.from_numpy(local_img)

        return img, local_img, label_tensor