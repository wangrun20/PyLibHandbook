import pandas as pd


def generate_excel(data=None, path='output.xlsx'):
    if data is None:
        data = {'列1标题': ['数据1', 1.6, 16],
                '列2标题': ['数据2', 1.7, 48],
                '列3标题': ['数据3', 5.4, 45.8]}
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


if __name__ == '__main__':
    generate_excel()
