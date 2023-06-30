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

# from robogym.envs.push.push_env import make_env
import gym

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="Reach")
parser.add_argument("--obs_size", default = 13, type = int)
parser.add_argument("--action_size", default = 4, type = int)
parser.add_argument("--feature_dims", default = 128, type = int)
parser.add_argument("--goal_size", default = 3, type = int)
parser.add_argument("--device", default="auto", type = str)
parser.add_argument("--min_action", default = -1., type = float)
parser.add_argument("--max_action", default = 1., type = float)
parser.add_argument("--max_episode_steps", default = 50, type = int)
parser.add_argument("--train_freq", default = 10, type = int)
parser.add_argument("--learning_starts", default = 50, type = int)
parser.add_argument("--save_interval", default = 50, type = int)
parser.add_argument("--train_cycle", default = 2, type = int)
parser.add_argument("--gradient_steps", default = 5, type = int)
parser.add_argument("--batch_size", default = 128, type = int)
parser.add_argument("--total_episodes", default = 1e6, type = int)
parser.add_argument("--eval_freq", default = 50, type = int)
parser.add_argument("--num_eval_episode", default = 10, type = int)
parser.add_argument("--relative_goal", default = True, type = bool)
args = parser.parse_args()


ACTION_SCALE = 1.
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
    gradient_steps = args.gradient_steps
    batch_size = args.batch_size
    total_episodes = args.total_episodes
    eval_freq = args.eval_freq
    num_eval_episode = args.num_eval_episode

    device = get_device(args.device)
    # save dir
    # save_dir = os.path.join(os.environ["VISUAL_PUSHING_HOME"], "log_files/her")
    # train_name = time.strftime("%m_%d-%H_%M", time.gmtime())
    # os.makedirs(os.path.join(save_dir, train_name), exist_ok=True)
    # save_path = os.path.join(save_dir, train_name)
    # model_path = os.path.join(save_path, 'her_models')
    # os.makedirs(model_path, exist_ok=True)
    # writer = SummaryWriter(save_path)
    # agent and env
    # env = make_env()
    agent = HER(
        observation_space,
        action_space,
        goal_space,
        feature_dims,
        min_action,
        max_action,
        max_episode_steps,
        train_freq,
        train_cycle,
        save_interval = args.save_interval,
        learning_starts = args.learning_starts,
        device = device,
        relative_goal = args.relative_goal,
    )
    # TODO load model

    # test
    episode = 0
    success_stats = []
    while episode < 100:
        env = gym.make('FetchReach-v1')
        obs_dict = env.reset()
        observation = np.empty(agent.dims['buffer_obs_size'], np.float32)
        achieved_goal = np.empty(agent.dims['goal'], np.float32)
        desired_goal = np.empty(agent.dims['goal'], np.float32)
        observation[:] = obs_dict['observation']
        achieved_goal[:] = obs_dict['achieved_goal']
        desired_goal[:] = obs_dict['desired_goal']

        obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(agent.max_episode_steps):
                env.render()
                # embed();exit()
                observation_new = np.empty(agent.dims['buffer_obs_size'], np.float32)
                achieved_goal_new = np.empty(agent.dims['goal'], np.float32)
                # success = np.zeros(1)

                # step env
                action = agent._sample_action(observation, achieved_goal,
                                             desired_goal, test = True)  # action is squashed to [-1, 1] by tanh function
                print(f"action: {action}")
                obs_dict_new, reward, done, _ = env.step(ACTION_SCALE * action)
                observation_new[:] = obs_dict_new['observation']
                achieved_goal_new[:] = obs_dict_new['achieved_goal']
                success = np.array(obs_dict_new['is_success'])

                # store transitions
                dones.append(done)
                obs.append(observation.copy())
                a_goals.append(achieved_goal.copy())
                acts.append(action.copy())
                d_goals.append(desired_goal.copy())
                successes.append(success.copy())

                # update states
                observation[:] = observation_new.copy()
                achieved_goal[:] = achieved_goal_new.copy()

            obs.append(observation.copy())
            a_goals.append(achieved_goal.copy())

            episode_transition = dict(
                o = np.array(obs).copy(),
                u = np.array(acts).copy(),
                g = np.array(d_goals).copy(),
                ag = np.array(a_goals).copy())
            # stats
            episode += 1
            agent.num_collected_episodes += 1
            success_stats.append(successes[-1])
            # add transition to replay buffer
            agent.rollout_buffer.add_episode_transitions(episode_transition)
            episode += 1
            env.close()

if __name__ == '__main__':
    main()