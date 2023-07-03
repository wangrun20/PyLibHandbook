import wandb


def main():
    """
    机器第一次运行wandb时，可能需要登录。建议从终端运行本脚本，即运行python use_wandb.py，再选择Use an existing W&B account，
        然后输入你的wandb账户的API key（在https://wandb.ai/settings里面找），即可。
        你的API key会被保存到~/.netrc这样的路径中，下次运行wandb时会自动读取，自动登录。
    wandb会将数据同时保存在本地（./wandb文件夹）和云端（wandb网站上）。
    """
    wandb.init(project='MyProject', name='go for it')
    for i in range(100):
        wandb.log({'iter': i + 1,
                   'epoch': (i // 10) + 1})


if __name__ == '__main__':
    main()
