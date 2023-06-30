import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from IPython import embed
import argparse
import os
import time
from IPython import embed

from VisualRL.rllib.her.her import HER
from VisualRL.rllib.common.utils import get_device, set_seed_everywhere

from robogym.envs.push.push_env import make_env
import gym

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="YCB-Pushing")
parser.add_argument("--obs_size", default = 12, type = int)
parser.add_argument("--action_size", default = 2, type = int)
parser.add_argument("--feature_dims", default = 128, type = int)
parser.add_argument("--goal_size", default = 3, type = int)
parser.add_argument("--device", default="auto", type = str)
parser.add_argument("--net_class", default="Flatten", type = str)
parser.add_argument("--min_action", default = -1., type = float)
parser.add_argument("--max_action", default = 1., type = float)
parser.add_argument("--max_episode_steps", default = 50, type = int)
parser.add_argument("--train_freq", default = 1, type = int)
parser.add_argument("--learning_starts", default = 2, type = int)
parser.add_argument("--learning_rate", default = 0.0003, type = float)
parser.add_argument("--save_interval", default = 100, type = int)
parser.add_argument("--step", default = 35000, type = int)
parser.add_argument("--train_cycle", default = 1, type = int)
parser.add_argument("--gradient_steps", default = 50, type = int)
parser.add_argument("--batch_size", default = 256, type = int)
parser.add_argument("--total_episodes", default = 1e6, type = int)
parser.add_argument("--eval_freq", default = 50, type = int)
parser.add_argument("--num_eval_episode", default = 10, type = int)
parser.add_argument("--relative_goal", action = "store_false")
parser.add_argument("--mp", action = "store_true")
parser.add_argument("--seed", default = None, type = int)
parser.add_argument("--load_weights", default=0, type=int)

args = parser.parse_args()
args.load_weights = 1

WEIGHT_PATH = "/homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_16-10_53/her_models"
ACTION_SCALE = 0.7
def main():
    observation_space = args.obs_size
    action_space = args.action_size
    goal_space = args.goal_size
    feature_dims = args.feature_dims
    min_action = args.min_action
    max_action = args.max_action
    max_episode_steps = args.max_episode_steps
    train_freq = args.train_freq
    # embed();exit()
    train_cycle = args.train_cycle

    device = get_device(args.device)
    # embed();exit()
    env = make_env()
    # env = gym.make("FetchPush-v1")
    agent = HER(
        observation_space,
        action_space,
        goal_space,
        env,
        feature_dims,
        min_action,
        max_action,
        max_episode_steps,
        train_freq,
        train_cycle,
        net_class = args.net_class,
        gradient_steps=args.gradient_steps,
        save_interval = args.save_interval,
        learning_starts = args.learning_starts,
        learning_rate = args.learning_rate,
        device = device,
        relative_goal = args.relative_goal,
        batch_size = args.batch_size,
    )
    # TODO load model
    if args.load_weights:
        print("loading model ...")
        agent.load(WEIGHT_PATH, args.step, map_location='cuda:1')


    # test
    episode = 0
    success_stats = []
    while episode < 4000:
        obs_dict = env.reset()
        start_time = time.time()
        observation = np.empty(agent.dims['buffer_obs_size'], np.float32)
        achieved_goal = np.empty(agent.dims['goal'], np.float32)
        desired_goal = np.empty(agent.dims['goal'], np.float32)
        observation[:] = obs_dict['observation']
        achieved_goal[:] = obs_dict['achieved_goal']
        desired_goal[:] = obs_dict['desired_goal']
        # while time.time() - start_time < 1.5:
        #     env.render()

        obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(10):
                # embed();exit()
                name = '/homeL/cong/HitLyn/Visual-Pushing/images/all_objects/' + "{:0>5d}.png".format(10 * episode + t)

                observation_new = np.empty(agent.dims['buffer_obs_size'], np.float32)
                achieved_goal_new = np.empty(agent.dims['goal'], np.float32)
                # success = np.zeros(1)

                # step env
                action = agent._select_action(observation, achieved_goal,
                                             desired_goal)  # action is squashed to [-1, 1] by tanh function
                print(f"action: {t}")
                obs_dict_new, reward, done, _ = env.step(ACTION_SCALE * action)
                observation_new[:] = obs_dict_new['observation']
                achieved_goal_new[:] = obs_dict_new['achieved_goal']
                success = np.array(obs_dict_new['is_success'])

                # update states
                observation[:] = observation_new.copy()
                achieved_goal[:] = achieved_goal_new.copy()
                # env.render()
                with env.mujoco_simulation.hide_target():
                    array = env.render(mode="rgb_array")
                plt.imsave(name, array, format='png')

            episode += 1
            # add transition to replay buffer
            # env.close()

if __name__ == '__main__':
    from mujoco_py import GlfwContext
    import matplotlib.pyplot as plt
    GlfwContext(offscreen=True)  # Create a window to init GLFW.
    main()