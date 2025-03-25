import cv2
import os

# 设置输入文件夹路径
folder1 = './Alldata/test/blended0'  # 替换为第一个文件夹路径
#folder2 = './Alldata/test/trans'  # dahaze
folder2 = './Alldata/testrdnet'   #rdnet
#output_folder = './Alldata/test/trans'  # 替换为输出文件夹路径
output_folder = './Alldata/testrdnet'
# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历第一个文件夹中的所有jpg文件
for filename in os.listdir(folder1):
    if filename.endswith('.jpg'):
        # 读取第一个文件夹中的图像
        img1_path = os.path.join(folder1, filename)
        img1 = cv2.imread(img1_path)

        # 获取第一个图像的尺寸
        h1, w1, _ = img1.shape

        # 读取第二个文件夹中的对应图像
        img2_path = os.path.join(folder2, filename)
        img2 = cv2.imread(img2_path)

        # 确保第二个图像存在
        if img2 is not None:
            # 从第二个图像中裁剪左上角与第一个图像相同大小的块
            cropped_img2 = img2[0:h1, 0:w1]

            # 保存裁剪后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img2)

            print(f'Processed: {filename}')
        else:
            print(f'Warning: {filename} not found in the second folder.')

print('All images have been processed.')
