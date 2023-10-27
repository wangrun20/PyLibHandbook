import os
import re


def find_files_with_x_in_name(folder_path):
    # 定义匹配包含"(x)"（x为数字）的正则表达式模式
    pattern = re.compile(r'\(\d+\)')

    # 获取文件夹中所有文件和子文件夹
    items = os.listdir(folder_path)

    for item in items:
        item_path = os.path.join(folder_path, item)

        # 如果是文件夹，递归调用该函数
        if os.path.isdir(item_path):
            find_files_with_x_in_name(item_path)
        else:
            # 使用正则表达式查找文件名中是否包含"(x)"
            if re.search(pattern, item):
                print(item_path)


if __name__ == '__main__':
    folder_to_search = "C:/Mine/自动化系学生科协2023-2024"
    find_files_with_x_in_name(folder_to_search)
