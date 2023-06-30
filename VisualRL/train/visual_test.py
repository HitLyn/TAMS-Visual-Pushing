import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import time
from IPython import embed
import cv2

from VisualRL.rllib.her.her import HER
from VisualRL.rllib.common.utils import get_device, set_seed_everywhere

from robogym.envs.push.visual_pushing import make_env
# from robogym.envs.push.push_env import make_env
# from robogym.envs.push.push_env import make_env
import gym

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW
# weights can be used:
# /homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_26-14_39/her_models, step 46700, vae:/homeL/cong/HitLyn/Visual-Pushing/results/vae/4/vae_model step = 100, latent_space:4, only 1 object
# /homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_28-16_10/her_models, step 46300, vae:/homeL/cong/HitLyn/Visual-Pushing/results/vae/4/vae_model step = 80, latent_space = 4, 4 objects
# /homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_30-14_25/her_models, step 51000, vae:/homeL/cong/HitLyn/Visual-Pushing/results/vae/04_30-13_51/vae_model, step = 100, latent_space = 4, all objects
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="YCB-Pushing")
parser.add_argument("--obs_size", default = 11, type = int)
parser.add_argument("--action_size", default = 2, type = int)
parser.add_argument("--feature_dims", default = 128, type = int)
parser.add_argument("--goal_size", default = 4, type = int)
parser.add_argument("--device", default="auto", type = str)
parser.add_argument("--net_class", default="Flatten", type = str)
parser.add_argument("--min_action", default = -1., type = float)
parser.add_argument("--max_action", default = 1., type = float)
parser.add_argument("--max_episode_steps", default = 50, type = int)
parser.add_argument("--train_freq", default = 1, type = int)
parser.add_argument("--learning_starts", default = 2, type = int)
parser.add_argument("--learning_rate", default = 0.0003, type = float)
parser.add_argument("--save_interval", default = 100, type = int)
parser.add_argument("--step", default = 51000, type = int)
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

WEIGHT_PATH = "/homeL/cong/HitLyn/Visual-Pushing/log_files/her/04_30-14_25/her_models"
# WEIGHT_PATH = "/homeL/cong/HitLyn/Visual-Pushing/log_files/her/07_05-08_40Order1/her_models"
ACTION_SCALE = 0.4
N = 50

EPISODE_STEP = 40
def main():
    observation_space = args.obs_size
    action_space = args.action_size
    goal_space = args.goal_size
    feature_dims = args.feature_dims
    min_action = args.min_action
    max_action = args.max_action
    max_episode_steps = args.max_episode_steps
    train_freq = args.train_freq
    train_cycle = args.train_cycle

    device = get_device(args.device)
    # embed();exit()
    env = make_env()
    env.reset()
    env.render()
    # env = gym.make("FetchPush-v1")
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
        env = env,
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
    # video record
    video_bottom = cv2.VideoWriter('/homeL/cong/Videos/push/saved/new_model/bottom_f04_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (720, 720), True)
    video_front = cv2.VideoWriter('/homeL/cong/Videos/push/saved/new_model/front_f04_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (720, 720), True)


    # data record
    # pusher trajectory, object trajectory, goal position (x,y,error)
    trajectory_pusher_record = np.zeros([N, EPISODE_STEP, 2])
    trajectory_object_record = np.zeros([N, EPISODE_STEP, 2])
    goal_pos_record = np.zeros([N, 3])


    while episode < N:
        obs_dict = env.reset()
        start_time = time.time()
        observation = np.empty(agent.dims['buffer_obs_size'], np.float32)
        achieved_goal = np.empty(agent.dims['goal'], np.float32)
        desired_goal = np.empty(agent.dims['goal'], np.float32)
        observation[:] = obs_dict['observation']
        achieved_goal[:] = obs_dict['achieved_goal']
        desired_goal[:] = obs_dict['desired_goal']
        # while time.time() - start_time < 2.5:
        #     # with env.mujoco_simulation.hide_target():
        #     env.render()

        obs, a_goals, acts, d_goals, successes, dones = [], [], [], [], [], []
        with torch.no_grad():
            for t in range(EPISODE_STEP):
                observation_new = np.empty(agent.dims['buffer_obs_size'], np.float32)
                achieved_goal_new = np.empty(agent.dims['goal'], np.float32)
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
                with env.mujoco_simulation.turn_targets_blue():
                    bottom_frame = env.sim.render(width = 720, height = 720, camera_name = 'phys_checks_cam')
                    bottom_frame = cv2.flip(cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2RGB),0)
                # bottom_frame = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2RGB)
                    front_frame = env.sim.render(width = 720, height = 720, camera_name = 'vision_cam_front')
                    front_frame = cv2.flip(cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB), 0)
                # front_frame = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                video_bottom.write(bottom_frame)
                video_front.write(front_frame)

                # with env.mujoco_simulation.hide_robot():
                    # env.render()
                    # with env.mujoco_simulation.hide_target():
                    #     array = env.render(mode="rgb_array")
                        # plt.imsave(name, array, format='png')
                trajectory_pusher_record[episode, t] = obs_dict_new["gripper_pos"].squeeze()[:2]
                trajectory_object_record[episode, t] = obs_dict_new["achieved_goal_gt"][:2]

            goal_pos_record[episode, :2] = obs_dict["desired_goal_gt"][:2]
            goal_pos_record[episode, 2] = np.linalg.norm(obs_dict_new["achieved_goal_gt"][:2] - obs_dict["desired_goal_gt"][:2])
            # exit()
            episode += 1
            print("episode: ", episode)


            # add transition to replay buffer
            # env.close()
    video_bottom.release()
    video_front.release()
    # np.save('../../data/pusher_trajectory_', trajectory_pusher_record)
    # np.save('../../data/object_trajectory_', trajectory_object_record)
    # np.save('../../data/goal', goal_pos_record)

if __name__ == '__main__':
    main()
