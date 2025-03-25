
import os
import cv2
import glob
import numpy as np

input_dir = 'data_low_light/Input/'
gt_dir = 'data_low_light/GT/'

output_input_dir = 'data_low_light_patch_512/input_dir/'
output_gt_dir = 'data_low_light_patch_512/gt_dir/'
output_input_val_dir = 'data_low_light_patch_512/input_val_dir/'
output_gt_val_dir = 'data_low_light_patch_512/gt_val_dir/'

if not os.path.isdir(output_input_dir):
    os.makedirs(output_input_dir)
if not os.path.isdir(output_gt_dir):
    os.makedirs(output_gt_dir)
if not os.path.isdir(output_input_val_dir):
    os.makedirs(output_input_val_dir)
if not os.path.isdir(output_gt_val_dir):
    os.makedirs(output_gt_val_dir)

test_fns = glob.glob(input_dir + '*.png')
test_ids = [os.path.basename(test_fn) for test_fn in test_fns]
# test_ids = test_ids[:100]
pad = 512
pad_h = 256
n = 0
m = 0

for k in range(len(test_ids)):
    print(k)

    in_path = input_dir + test_ids[k]
    gt_path = gt_dir + test_ids[k]

    input_data = cv2.imread(in_path)
    gt_data = cv2.imread(gt_path)

    h_pad = 0
    w_pad = 0

    h1,w1,c1= input_data.shape

    if (h1 % pad_h) !=0:
        h_pad = pad_h - (h1 % pad_h)
    if (w1 % pad_h) !=0:
        w_pad = pad_h - (w1 % pad_h)

    input_data2 = cv2.copyMakeBorder(input_data, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    gt_data2 = cv2.copyMakeBorder(gt_data, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

    h1,w1,c1= input_data2.shape

    hh = (h1 - pad_h) // pad_h
    ww = (w1 - pad_h) // pad_h

    if k % 10 == 0:
        for k1 in range(hh):
            for k2 in range(ww):
                in_pic = input_data2[pad_h*k1:pad_h*k1 + pad, pad_h*k2:pad_h*k2 + pad]
                gt_pic = gt_data2[pad_h*k1:pad_h*k1 + pad, pad_h*k2:pad_h*k2 + pad]
                # m = np.mean(pic)

                # if m > 192:
                cv2.imwrite(output_input_val_dir + 'test_%07d.png'%n, in_pic)
                cv2.imwrite(output_gt_val_dir + 'test_%07d.png'%n, gt_pic)
                n = n + 1
    else:
        for k1 in range(hh):
            for k2 in range(ww):
                in_pic = input_data2[pad_h*k1:pad_h*k1 + pad, pad_h*k2:pad_h*k2 + pad]
                gt_pic = gt_data2[pad_h*k1:pad_h*k1 + pad, pad_h*k2:pad_h*k2 + pad]
                # m = np.mean(pic)

                # if m > 192:
                cv2.imwrite(output_input_dir + 'test_%07d.png'%m, in_pic)
                cv2.imwrite(output_gt_dir + 'test_%07d.png'%m, gt_pic)
                m = m + 1

print(n)
print(m)