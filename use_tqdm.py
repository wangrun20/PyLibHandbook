import time

from tqdm import tqdm


def main():
    length = 100
    with tqdm(desc=f'processing', total=length, unit='items') as pbar:
        for i in range(length):
            time.sleep(0.1)
            pbar.set_postfix({'second': (i + 1) * 0.1})
            pbar.update(1)


if __name__ == '__main__':
    main()
