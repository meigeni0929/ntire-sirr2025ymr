import os
import random
import shutil
from datetime import datetime


def validate_paths(source_dir, target_dir):
    """路径验证函数"""
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源文件夹不存在: {source_dir}")
    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        raise ValueError("目标文件夹不能与源文件夹相同")


def get_image_files(source_dir):
    """获取所有图片文件（带完整路径）"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    return [os.path.join(source_dir, f) for f in os.listdir(source_dir)
            if os.path.splitext(f)[1].lower() in valid_ext]


def backup_files(file_list, target_dir):
    """带时间戳的备份机制"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(target_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    for file_path in file_list:
        try:
            shutil.copy2(file_path, backup_dir)  # 保留文件元数据
        except Exception as e:
            print(f"复制失败 {os.path.basename(file_path)}: {str(e)}")
            raise
    return backup_dir


def user_confirmation(file_list):
    """交互式确认机制"""
    print(f"即将处理 {len(file_list)} 个文件")
    print("示例文件：")
    for f in file_list[:3]:
        print(f"  {os.path.basename(f)}")
    return input("确认操作？(y/n): ").lower() == 'y'


def safe_file_operation(source_dir, target_dir, num=30):
    # 路径安全检查
    validate_paths(source_dir, target_dir)

    # 获取图片文件
    all_images = get_image_files(source_dir)
    if len(all_images) < num:
        raise ValueError(f"图片数量不足，需要 {num} 张，实际找到 {len(all_images)} 张")

    # 随机选择文件
    selected = random.sample(all_images, num)
    selected_basenames = [os.path.basename(f) for f in selected]

    # 用户确认
    if not user_confirmation(selected):
        print("操作已取消")
        return

    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 第一步：备份文件
        backup_path = backup_files(selected, target_dir)
        print(f"文件已备份至：{backup_path}")

        # 第二步：删除源文件
        for file_path in selected:
            os.remove(file_path)
            print(f"已删除源文件：{os.path.basename(file_path)}")

    except Exception as e:
        print(f"操作中断，已备份文件保留在 {backup_path}")
        raise RuntimeError("操作未完成，部分文件可能已处理") from e


if __name__ == "__main__":
    # 配置参数（使用前务必修改）
    SOURCE_DIR = "/data_nvme/ymr/SIRR/train_800/input"  # 源图片文件夹
    TARGET_DIR = "/data_nvme/ymr/SIRR/val/input"  # 目标保存文件夹

    # 执行安全操作
    try:
        safe_file_operation(SOURCE_DIR, TARGET_DIR)
        print("操作成功完成")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 安全提示
        print("\n重要提示：")
        print("1. 源文件夹文件已被永久删除")
        print("2. 请检查备份文件夹确认文件完整性")
        print("3. 建议定期备份重要数据")
