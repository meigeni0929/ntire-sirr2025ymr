from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from os import listdir
import cv2
import numpy as np
import torch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class dehaze_train_dataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.filenames = [x for x in listdir(input_dir) if is_image_file(x)]

        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.file_len = len(self.filenames)


    def __getitem__(self, index):
        # print(self.input_dir + self.filenames[index])
        in_image_cv2 = cv2.imread(self.input_dir + self.filenames[index])[:,:,::-1] / 255.
        gt_image_cv2 = cv2.imread(self.gt_dir + self.filenames[index])[:,:,::-1] / 255.

        h,w,c = in_image_cv2.shape

        h_n = np.random.randint(513,np.max([514,h]))
        w_n = np.random.randint(513,np.max([514,w]))

        h_f = np.random.randint(h_n - 512)
        w_f = np.random.randint(w_n - 512)

        in_image_cv2 = cv2.resize(in_image_cv2,(w_n,h_n))
        gt_image_cv2 = cv2.resize(gt_image_cv2,(w_n,h_n))

        in_image_cv2_512 = in_image_cv2[h_f:h_f+512, w_f:w_f+512, :]
        gt_image_cv2_512 = gt_image_cv2[h_f:h_f+512, w_f:w_f+512, :]

        ind = np.random.randint(8)
        if ind > 3:
            in_image_cv2_512 = in_image_cv2_512[:,::-1,:]
            gt_image_cv2_512 = gt_image_cv2_512[:,::-1,:]

        in_image_cv2_512 = np.rot90(in_image_cv2_512, ind%4)
        gt_image_cv2_512 = np.rot90(gt_image_cv2_512, ind%4)

        in_image_cv2_512 = np.ascontiguousarray(in_image_cv2_512)
        gt_image_cv2_512 = np.ascontiguousarray(gt_image_cv2_512)

        pic_torch = torch.from_numpy(in_image_cv2_512).permute(2,0,1)

        gt_torch = torch.from_numpy(gt_image_cv2_512).permute(2,0,1)

        return pic_torch, gt_torch
    

    def __len__(self):
        return self.file_len





