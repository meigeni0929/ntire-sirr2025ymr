import cv2
import os
import numpy as np

input_folder = './Alldata/test/blendede0'  # 
output_folder = './Alldata/test/blendede'  # 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

target_size = (1440, 1440)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
       
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        
        h, w, _ = img.shape

        
        pad_height = max(0, target_size[0] - h)
        pad_width = max(0, target_size[1] - w)

        # 进行反射填充
        # cv2.copyMakeBorder的参数分别为：上、下、左、右、填充类型
        padded_img = cv2.copyMakeBorder(
            img,
            0, pad_height, 0, pad_width,
            cv2.BORDER_REFLECT
        )

        # 保存填充后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, padded_img)

        print(f'Processed: {filename}')

print('All images have been processed.')
