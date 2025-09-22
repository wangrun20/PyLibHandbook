from dataclasses import dataclass
from typing import Optional
from functools import partial

from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


@dataclass
class PPOConfig:
    # 训练配置
    num_iterations: int = 1500
    learning_rate: float = 1e-3
    num_steps: int = 24
    schedule: str = 'adaptive'

    # PPO超参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 5
    norm_adv: bool = False
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = 0.01

    # 计算得出的参数（运行时填充）
    batch_size: int = 0
    minibatch_size: int = 0


class DummyEnv(object):
    """用纯torch实现的简单2D质点运动环境，支持批处理"""
    def __init__(self, dt=0.01, num_envs=1024, device="cuda"):
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.state = torch.zeros(size=(self.num_envs, 4), device=self.device)  # 状态：[x, y, vx, vy]
        self.episode_length = torch.zeros(size=(self.num_envs,), device=self.device)
        self.episode_reward = torch.zeros(size=(self.num_envs,), device=self.device)
        self.last_episode_length = torch.zeros(size=(self.num_envs,), device=self.device)
        self.last_episode_reward = torch.zeros(size=(self.num_envs,), device=self.device)
        self.infos = {}

        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.single_action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)  # 动作：[ax, ay]

    def reset(self):
        self.state = torch.randn(size=(self.num_envs, 4), device=self.device) * 2.0
        self.episode_length = torch.zeros(size=(self.num_envs,), device=self.device)
        next_obs = self.state.clone()
        return next_obs, self.infos

    def step(self, action):
        pos = self.state[:, :2]
        vel = self.state[:, 2:]
        acc = action
        new_vel = vel + acc * self.dt
        new_pos = pos + vel * self.dt + acc * (self.dt ** 2)
        self.state = torch.cat([new_pos, new_vel], dim=1)

        reward = torch.exp(- torch.linalg.norm(new_pos, dim=1))  # 简单奖励：距离原点越近奖励越高

        self.episode_length += 1
        self.episode_reward += reward

        terminations = torch.norm(new_pos, dim=1) > 5.0  # 终止条件：距离原点太远或步数太多
        truncations = torch.greater_equal(self.episode_length, 1000)
        dones = torch.logical_or(terminations, truncations)

        self.last_episode_length = torch.where(dones, self.episode_length, self.last_episode_length)
        self.last_episode_reward = torch.where(dones, self.episode_reward, self.last_episode_reward)

        self.infos = {
            'episode_length': torch.mean(self.last_episode_length).item(),
            'episode_reward': torch.mean(self.last_episode_reward).item(),
            'single_reward_distance': torch.mean(reward).item()
        }

        # auto reset
        self.state = torch.where(
            dones[:, None], torch.randn(size=self.state.shape, device=self.device) * 2.0, self.state)
        self.episode_length = torch.where(dones, 0, self.episode_length)
        self.episode_reward = torch.where(dones, 0, self.episode_reward)
        next_obs = self.state.clone()

        return next_obs, reward, terminations, truncations, self.infos.copy()


class DummyEnvRSL(VecEnv):
    """将DummyEnv适配为rsl_rl兼容的环境"""
    def __init__(self, dt=0.01, num_envs=1024, device="cuda"):
        self.dummy_env = DummyEnv(dt=dt, num_envs=num_envs, device=device)

        self.num_envs = num_envs
        self.num_actions = 2
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(size=(self.num_envs,), device=device)
        self.device = torch.device(device)

    def reset(self):
        next_obs, infos = self.dummy_env.reset()
        extras = {'observations': {}}
        return next_obs, extras

    def step(self, actions):
        next_obs, reward, terminations, truncations, infos = self.dummy_env.step(actions)
        dones = torch.logical_or(truncations, terminations)
        return next_obs, reward, dones, {'log': infos}

    def get_observations(self):
        obs = self.dummy_env.state.clone()
        extras = {'observations': {}}
        return obs, extras


def eval_and_plot(get_action_func, dummy_env: DummyEnv, num_episodes=5):
    """评估训练后的智能体并绘制轨迹"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(range(num_episodes))

    for episode in range(num_episodes):
        obs, _ = dummy_env.reset()
        obs = obs[0]  # 只用第一个环境
        trajectory = [obs[:2].cpu().numpy()]

        for step in range(1000):
            action = get_action_func(obs)
            with torch.no_grad():
                obs, reward, term, trunc, _ = dummy_env.step(action)
            if term[0] or trunc[0]:
                break
            obs = obs[0]  # 只用第一个环境
            trajectory.append(obs[:2].cpu().numpy())

        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1],
                 color=colors[episode], alpha=0.7, linewidth=2,
                 label=f'Episode {episode + 1}')
        plt.scatter(trajectory[0, 0], trajectory[0, 1],
                    color=colors[episode], s=100, marker='o', edgecolors='black')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1],
                    color=colors[episode], s=100, marker='s', edgecolors='black')

    plt.scatter(0, 0, color='red', s=200, marker='*', label='Target')
    circle = plt.Circle((0, 0), 5, fill=False, color='red', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, 64)),
            nn.ELU(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_mean(self, obs):
        action_mean = self.actor_mean(obs)
        return action_mean

    def get_action_std(self):
        action_std = torch.exp(self.actor_logstd)
        return action_std

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


class MyPPO(object):
    def __init__(self, env, config: PPOConfig):
        self.env = env
        self.num_envs = self.env.num_envs

        self.config = config
        self.config.batch_size = int(self.num_envs * self.config.num_steps)
        self.config.minibatch_size = int(self.config.batch_size // self.config.num_minibatches)

        self.device = self.env.device

        assert isinstance(self.env.single_observation_space, spaces.Box)
        assert isinstance(self.env.single_action_space, spaces.Box)
        assert len(self.env.single_observation_space.shape) == 1
        assert len(self.env.single_action_space.shape) == 1
        self.obs_dim = self.env.single_observation_space.shape
        self.act_dim = self.env.single_action_space.shape

        self.agent = Agent(self.obs_dim[0], self.act_dim[0]).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.config.learning_rate, eps=1e-5)
        self.learning_rate = self.config.learning_rate

        self.obs = torch.zeros((self.config.num_steps, self.num_envs) + self.obs_dim, device=self.device)
        self.actions = torch.zeros((self.config.num_steps, self.num_envs) + self.act_dim, device=self.device)
        self.logprobs = torch.zeros((self.config.num_steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.config.num_steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.config.num_steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.config.num_steps, self.num_envs), device=self.device)

        self.next_obs = torch.zeros((self.num_envs,) + self.obs_dim, device=self.device)
        self.next_done = torch.zeros((self.num_envs,), device=self.device)
        self.global_step = 0
        self.current_iteration = 0

    def reset(self):
        next_obs, _ = self.env.reset()
        self.next_obs = next_obs.clone()
        self.next_done = torch.zeros(self.num_envs, device=self.device)

    def train(self, callback=None):
        self.reset()
        all_infos = []

        for iteration in range(1, self.config.num_iterations + 1):
            train_info = self.train_step()
            all_infos.append(train_info)

            if callback:
                callback(train_info)

        return {'all_infos': all_infos}

    def train_step(self):
        self.current_iteration += 1

        rollout_info = self.collect_rollouts()

        advantages, returns = self.compute_gae()

        update_info = self.update_policy(advantages, returns)

        mean_std = torch.mean(self.agent.get_action_std().detach()).item()

        train_info = {'rollout_info': rollout_info,
                      'update_info': update_info,
                      'iteration': self.current_iteration,
                      'global_step': self.global_step,
                      'mean_std': mean_std,
                      'lr': self.learning_rate}

        return train_info

    def collect_rollouts(self):
        infos_list = []
        for i in range(0, self.config.num_steps):
            self.global_step += self.num_envs
            self.obs[i] = self.next_obs
            self.dones[i] = self.next_done
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                self.values[i] = value.flatten()
            self.actions[i] = action
            self.logprobs[i] = logprob

            next_obs, reward, terminations, truncations, infos = self.env.step(action)
            self.next_obs = next_obs.clone()
            self.rewards[i] = reward
            self.next_done = torch.logical_or(terminations, truncations).float()
            infos_list.append(infos)
        rollout_info = {}
        for key in infos_list[0].keys():
            rollout_info[key] = []
            for i in range(len(infos_list)):
                rollout_info[key].append(infos_list[i][key])
            rollout_info[key] = np.mean(rollout_info[key])
        return rollout_info

    def compute_gae(self):
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.gamma * nextvalues * nextnonterminal - self.values[t]
                lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + self.values
        return advantages, returns

    def update_policy(self, advantages, returns):
        b_obs = self.obs.reshape((-1,) + self.obs_dim)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        if self.config.target_kl is not None and self.config.schedule == "adaptive":
            with torch.no_grad():
                old_mu_batch = self.agent.get_action_mean(b_obs)
                old_sigma_batch = self.agent.get_action_std().expand_as(old_mu_batch)

        b_inds = np.arange(self.config.batch_size)
        clipfracs = []

        pg_loss = torch.tensor(float('nan'))
        v_loss = torch.tensor(float('nan'))
        entropy_loss = torch.tensor(float('nan'))
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 策略损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], min=-self.config.clip_coef, max=self.config.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                if self.config.target_kl is not None and self.config.schedule == "adaptive":
                    with torch.inference_mode():
                        mu_mb = self.agent.get_action_mean(b_obs[mb_inds])
                        sigma_mb = self.agent.get_action_std().expand_as(mu_mb)
                        old_mu_mb = old_mu_batch[mb_inds]
                        old_sigma_mb = old_sigma_batch[mb_inds]
                        kl = torch.sum(torch.log(sigma_mb / old_sigma_mb + 1.0e-5) +
                                       (torch.square(old_sigma_mb) + torch.square(old_mu_mb - mu_mb))
                                       / (2.0 * torch.square(sigma_mb)) - 0.5, dim=-1)
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.config.target_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < (self.config.target_kl / 2.0):
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break

        # 计算解释方差
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        update_info = {'policy_loss': pg_loss.item(),
                       'value_loss': v_loss.item(),
                       'entropy_loss': entropy_loss.item(),
                       'old_approx_kl': old_approx_kl.item(),
                       'approx_kl': approx_kl.item(),
                       'clipfrac': np.mean(clipfracs),
                       'explained_variance': explained_var,
                       'learning_rate': self.optimizer.param_groups[0]["lr"]}

        return update_info

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'current_iteration': self.current_iteration
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_iteration = checkpoint.get('current_iteration', 0)

    def get_action(self, obs, deterministic: bool = False):
        """获取动作（用于推理）"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.agent.actor_mean(obs)
            else:
                action, _, _, _ = self.agent.get_action_and_value(obs)
        return action


def using_my_ppo():
    env = DummyEnv()
    ppo_config = PPOConfig()
    ppo = MyPPO(env=env, config=ppo_config)

    def training_callback(info):
        print(f"================\n"
              f"Iter {info['iteration']}\n"
              f"Avg Length = {info['rollout_info']['episode_length']:.2f}\n"
              f"Avg Reward = {info['rollout_info']['episode_reward']:.2f}\n"
              f"Dis Reward = {info['rollout_info']['single_reward_distance']:.2f}\n"
              f"   Avg Std = {info['mean_std']:.3f}\n"
              f"        lr = {info['lr']:.3e}\n")

    print("开始训练...")
    ppo.train(callback=training_callback)

    eval_and_plot(partial(ppo.get_action, deterministic=True), DummyEnv(dt=env.dt, num_envs=1, device=env.device))


def using_rsl_rl():
    env = DummyEnvRSL()

    train_cfg = {
        'num_steps_per_env': 24,
        'save_interval': 100,
        'empirical_normalization': False,
        'policy': {
            'class_name': 'ActorCritic',
            'init_noise_std': 1.0,
            'noise_std_type': 'scalar',
            'actor_hidden_dims': [64, 64],
            'critic_hidden_dims': [64, 64],
            'activation': 'elu'
        },
        'algorithm': {
            'class_name': 'PPO',
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'learning_rate': 1e-3,
            'schedule': 'adaptive',
            'gamma': 0.99,
            'lam': 0.95,
            'entropy_coef': 0.01,
            'desired_kl': 0.01,
            'max_grad_norm': 1.0,
            'value_loss_coef': 1.0,
            'use_clipped_value_loss': True,
            'clip_param': 0.2,
            'normalize_advantage_per_mini_batch': False,
            'symmetry_cfg': None,
            'rnd_cfg': None
        }
    }

    runner = OnPolicyRunner(env=env,
                            train_cfg=train_cfg,
                            log_dir='exp',
                            device=env.device)

    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)

    eval_and_plot(runner.get_inference_policy(),
                  DummyEnv(dt=env.dummy_env.dt, num_envs=1, device=env.dummy_env.device))


if __name__ == "__main__":
    using_my_ppo()
    # using_rsl_rl()
