import configparser
import os


class PPOConfig:
    """
    PPO + GAE 超参数配置

    支持两种加载方式:
        1. 从 .ini 配置文件加载
        2. 直接通过构造函数传参覆盖
    """

    def __init__(self, config_path=None, **overrides):
        defaults = self._defaults()

        if config_path is not None and os.path.exists(config_path):
            file_config = self._load_from_ini(config_path)
            defaults.update(file_config)

        defaults.update(overrides)

        for key, value in defaults.items():
            setattr(self, key, value)

    @staticmethod
    def _defaults():
        return dict(
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            reward_gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,

            actor_hidden_size=128,
            critic_hidden_size=128,

            actor_lr=3e-4,
            critic_lr=3e-4,

            action_space_type='discrete',
            state_dim=25,
            action_dim=5,

            use_cuda=True,
            seed=42,

            max_episodes=10000,
            eval_interval=50,
            eval_episodes=10,
            save_interval=100,
        )

    @staticmethod
    def _load_from_ini(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        result = {}
        for section in config.sections():
            for key, value in config[section].items():
                try:
                    if '.' in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        result[key] = value.lower() == 'true'
                    else:
                        result[key] = value
        return result

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}