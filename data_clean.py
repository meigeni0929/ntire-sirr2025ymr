import os
import shutil
from pathlib import Path

def find_and_delete_duplicates(dir1, dir2):
    # 获取两个目录的图片文件名（包含扩展名）
    dir2_files = {f.name for f in Path(dir2).glob("*") if f.is_file()}
    dir1_files = [f for f in Path(dir1).glob("*") if f.is_file()]

    # 找出重复文件
    duplicates = [f for f in dir1_files if f.name in dir2_files]

    if not duplicates:
        print("没有发现重复文件")
        return

    # 显示检测结果
    print(f"发现 {len(duplicates)} 个重复文件：")
    for idx, file in enumerate(duplicates, 1):
        print(f"{idx}. {file.name}")

    # 确认删除
    confirm = input("\n确认要删除这些文件吗？(y/n): ").lower()
    if confirm == 'y':
        deleted_count = 0
        for file in duplicates:
            try:
                file.unlink()  # 删除文件
                deleted_count += 1
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除失败 [{file}]: {str(e)}")
        print(f"\n成功删除 {deleted_count}/{len(duplicates)} 个文件")
    else:
        print("操作已取消")

if __name__ == "__main__":
    dir1 = input("/data_nvme/ymr/SIRR/train_800/transmission_layer").strip()
    dir2 = input("/data_nvme/ymr/SIRR/val/transmission_layer").strip()

    try:
        find_and_delete_duplicates(dir1, dir2)
    except Exception as e:
        print(f"发生错误: {str(e)}")
