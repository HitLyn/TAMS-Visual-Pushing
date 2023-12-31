from stable_baselines3 import HER, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
import gym
from robogym.envs.push.push_env import make_env
model_class = SAC  # works also with SAC, DDPG and TD3
N_BITS = 50

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)
env = gym.make("FetchPush-v1")
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, online_sampling=online_sampling,
                        verbose=1, max_episode_length=max_episode_length)
# Train the model
model.learn(10000000000000)

# model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = HER.load('./her_bit_env', env=env)
#
# obs = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _ = env.step(action)
#
#     if done:
#         obs = env.reset()