import os

def delete_duplicate_images(folder1, folder2):
    # 定义图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    # 收集folder2中图片文件的主文件名（不区分大小写）
    folder2_names = set()
    for filename in os.listdir(folder2):
        if os.path.isfile(os.path.join(folder2, filename)):
            main_name, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions:
                folder2_names.add(main_name.lower())  # 不区分大小写

    # 遍历folder1并删除重复文件
    deleted = []
    errors = []
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        if os.path.isfile(file_path):
            main_name, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions:
                if main_name.lower() in folder2_names:
                    try:
                        os.remove(file_path)
                        deleted.append(filename)
                    except Exception as e:
                        errors.append(f"{filename}: {str(e)}")

    # 输出结果
    print(f"已删除 {len(deleted)} 个文件:")
    for name in deleted:
        print(f"• {name}")
    
    if errors:
        print("\n删除时发生错误:")
        for error in errors:
            print(f"• {error}")

# 使用示例
if __name__ == "__main__":
    folder1 = "/data_ssd/ymr/SIRR/train_800/transmission_layer"
    folder2 = "/data_ssd/ymr/SIRR/val/transmission_layer"

    # 验证路径有效性
    if not all(map(os.path.isdir, [folder1, folder2])):
        print("错误: 输入的路径不是有效文件夹")
        exit(1)

    delete_duplicate_images(folder1, folder2)
