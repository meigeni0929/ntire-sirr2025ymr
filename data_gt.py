import os
import shutil
from pathlib import Path
from datetime import datetime


def find_and_save_duplicates(source_dir1, source_dir2, output_dir):
    """
    查找两个源目录中的同名图片并保存到输出目录
    输出目录结构：
    output_dir/
        duplicates/
            source1/
                file1.jpg
                subdir/file2.png
            source2/
                file1.jpg
                subdir/file2.png
        report.txt
    """
    # 创建输出目录结构
    output_duplicates = Path(output_dir) / "duplicates"
    output_source1 = output_duplicates / "source1"
    output_source2 = output_duplicates / "source2"
    report_file = Path(output_dir) / "report.txt"

    # 初始化输出目录
    for d in [output_source1, output_source2]:
        d.mkdir(parents=True, exist_ok=True)

    # 构建文件名索引
    def build_index(folder):
        index = {}
        for root, _, files in os.walk(folder):
            for f in files:
                if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}:
                    rel_path = os.path.relpath(root, folder)
                    index.setdefault(f.lower(), []).append({
                        'abs_path': os.path.join(root, f),
                        'rel_path': rel_path
                    })
        return index

    # 构建两个源目录的索引
    index1 = build_index(source_dir1)
    index2 = build_index(source_dir2)

    # 查找重复文件名
    common_files = set(index1.keys()) & set(index2.keys())

    # 生成报告内容
    report_content = [
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"源目录1: {source_dir1}",
        f"源目录2: {source_dir2}",
        f"共发现 {len(common_files)} 个重复文件\n"
    ]

    # 复制文件并记录操作
    copied_files = []
    for filename in common_files:
        # 处理第一个源目录的文件
        for file_info in index1[filename]:
            src = file_info['abs_path']
            dest_dir = output_source1 / file_info['rel_path']
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            shutil.copy2(src, dest)
            copied_files.append(("source1", str(dest)))

        # 处理第二个源目录的文件
        for file_info in index2[filename]:
            src = file_info['abs_path']
            dest_dir = output_source2 / file_info['rel_path']
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / filename
            shutil.copy2(src, dest)
            copied_files.append(("source2", str(dest)))

        # 记录报告
        paths1 = [f"  - {info['abs_path']}" for info in index1[filename]]
        paths2 = [f"  - {info['abs_path']}" for info in index2[filename]]
        report_content.extend([
            f"文件名: {filename}",
            "源目录1出现位置:",
            *paths1,
            "源目录2出现位置:",
            *paths2,
            "-" * 50
        ])

    # 写入报告文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    return {
        'total_duplicates': len(common_files),
        'copied_files': copied_files,
        'report_path': str(report_file)
    }


if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    SOURCE_DIR1 = "/data_nvme/ymr/SIRR/val/input"
    SOURCE_DIR2 = "/data_nvme/ymr/SIRR/train_800/gt"
    OUTPUT_DIR = "/data_nvme/ymr/SIRR/val/gt"

    # 执行操作
    result = find_and_save_duplicates(SOURCE_DIR1, SOURCE_DIR2, OUTPUT_DIR)

    # 输出结果
    print(f"操作完成，共处理 {result['total_duplicates']} 个重复文件")
    print(f"详细报告见：{result['report_path']}")
    print("已复制的文件：")
    for source, path in result['copied_files'][:3]:  # 显示前3个示例
        print(f"  [{source}] {path}")
