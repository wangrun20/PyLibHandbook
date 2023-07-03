import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='conf', config_name='sample')
def main(cfg: DictConfig):
    print(type(cfg))
    print(cfg)
    yaml_str = OmegaConf.to_yaml(cfg)
    print(type(yaml_str))
    print(yaml_str)


if __name__ == "__main__":
    """
    learn more about Hydra at https://hydra.cc/docs/intro/
    从命令行运行脚本时，可以覆盖.yaml中的配置，如：python use_hydra.py env.task=racecar
    Hydra默认会将运行时产生的日志和config备份保存在./outputs文件夹中
    """
    main()
