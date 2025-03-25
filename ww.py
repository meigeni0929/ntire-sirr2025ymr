import os
from PIL import Image
import re


def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)

        # 验证是否为4位数字的文件夹
        if re.match(r'^test_\d{4}$', folder) and os.path.isdir(folder_path):
            src_path = os.path.join(folder_path, 'ytmt_ucs_sirs_t.png')

            if os.path.exists(src_path):
                try:
                    with Image.open(src_path) as img:
                        rgb_img = img.convert('RGB')
                        # 直接使用文件夹名称（如0001）作为文件名
                        dest_path = os.path.join(output_dir, f"{folder}.jpg")  # <-- 修改这里
                        rgb_img.save(dest_path, 'JPEG', quality=95)
                        print(f"转换成功: {folder} -> {folder}.jpg")
                except Exception as e:
                    print(f"处理失败 [{folder}]: {str(e)}")
            else:
                print(f"跳过 [{folder}]: 未找到图片")


if __name__ == '__main__':
    process_images("./Alldata/testout", "./Alldata/testrdnet")
    print("\n完成！输出路径:", os.path.abspath("./Alldata/testrdnet"))
