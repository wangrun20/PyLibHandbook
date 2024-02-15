import os


def rename_files_by_date(path):
    # 获取文件夹中所有文件的路径及最后修改时间
    files_with_dates = [(os.path.join(path, filename), os.path.getmtime(os.path.join(path, filename))) for
                        filename in os.listdir(path)]
    # 根据最后修改时间排序
    files_with_dates.sort(key=lambda x: x[1])

    # 确定数字位数
    num_digits = len(str(len(files_with_dates)))

    # 重命名文件
    for i, (file_path, _) in enumerate(files_with_dates):
        _, extension = os.path.splitext(file_path)
        new_filename = str(i + 1).zfill(num_digits) + extension
        new_file_path = os.path.join(path, new_filename)
        os.rename(file_path, new_file_path)


# 指定文件夹路径
folder_path = r"E:\Dirty\x6o论坛\[紧急企划] - 樱可 跳蛋"

# 调用函数
rename_files_by_date(folder_path)
