import os
from datetime import datetime
import matplotlib.pyplot as plt


def get_file_stats(path):
    # 初始化日期列表和文件路径字典
    dates = []
    file_paths = {}

    # 递归遍历文件夹
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                # 获取文件的修改日期
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                dates.append(mod_time)
                file_paths[mod_time] = file_path
            except OSError:
                pass

    return dates, file_paths


def main():
    folder_path = input("请输入要遍历的文件夹路径: ")
    dates, file_paths = get_file_stats(folder_path)

    if not dates:
        print("未找到任何文件的修改日期。")
        return

    # 输出修改日期最早和最晚的10个文件的路径
    sorted_dates = sorted(file_paths.keys())
    print("\n修改日期最早的10个文件:")
    for date in sorted_dates[:10]:
        print(f"{date.strftime('%Y-%m-%d %H:%M:%S')}: {file_paths[date]}")

    print("\n修改日期最晚的10个文件:")
    for date in sorted_dates[-10:]:
        print(f"{date.strftime('%Y-%m-%d %H:%M:%S')}: {file_paths[date]}")

    # 绘制频率分布直方图
    plt.hist(dates, bins=30, edgecolor='k')
    plt.xlabel('Modification date')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
