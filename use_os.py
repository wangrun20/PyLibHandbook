import os


def traverse_files(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            print(item_path)
            # do something
        elif os.path.isdir(item_path):
            traverse_files(item_path)


def main():
    folder_path = './'
    traverse_files(folder_path)


if __name__ == '__main__':
    main()
