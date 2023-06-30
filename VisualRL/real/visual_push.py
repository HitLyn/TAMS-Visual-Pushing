#!/usr/bin/env python

import torch
from torchvision import transforms
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64MultiArray
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import ros_numpy
from VisualRL.rllib.her.her import HER
from VisualRL.vae.model import VAE
from VisualRL.rllib.common.utils import get_device

from IPython import embed

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
parser.add_argument("--load_weights", default=1, type=int)
args = parser.parse_args()

WEIGHT_PATH = "/homeL/cong_pushing/HitLyn/Visual-Pushing/log_files/her/04_30-14_25/her_models"
ACTION_SCALE = 0.5
VELOCITY_SCALE = 0.2
# STEP = 51000

IMAGE_PATH = '/homeL/pushing/pushing_masks'


class VisualPushing:
    def __init__(self, device = None):
        self.step_ = 0
        self.goal_latent = None
        self.goal_mask = None
        self.vel_pub = rospy.Publisher('/dynamic_pushing/velocity', geometry_msgs.msg.Vector3, queue_size=1)
        self.goal_publisher = rospy.Publisher('/pushing/goal_recon', Image, queue_size = 2)
        self.mask_publisher = rospy.Publisher('/pushing/mask', Image, queue_size=2)
        self.image_recon_publisher = rospy.Publisher('/pushing/mask_recon', Image, queue_size=2)
        self.image_crop_publisher = rospy.Publisher('/pushing/image_crop', Image, queue_size=2)
        self.image_crop_segmentation_publisher = rospy.Publisher('/pushing/image_crop_hsv', Image, queue_size=2)
        self.overall_publisher = rospy.Publisher('/pushing/overall', Image, queue_size=2)
        # update pusher tf
        self.gripper_pos = np.zeros(3)
        rospy.Subscriber("/tf_pusher", Float64MultiArray, self.pusher_tf_update)

        # self.bridge = CvBridge()
        self.device = torch.device('cuda:0') if device == None else torch.device('cuda:1')
        # tf
        # motion range
        self.x_range = np.array([0.43, 0.90])
        self.y_range = np.array([0.35, 1.10])
        # goal pose initialization
        self.vae = VAE(device = self.device, image_channels = 1, h_dim = 1024, z_dim = 4)
        self.vae.load("/homeL/cong_pushing/HitLyn/Visual-Pushing/results/vae/04_30-13_51/vae_model", 100, map_location=self.device)  # latent space = 4
        self.goal_latent, self.goal_mask = self.initialize_goal()
        # self.goal_latent = np.array([ 1.4946128,  0.6009053,  5.4465885, -1.370616 ])



    def clip_action(self, gripper_pos, action):
        # x direction
        if gripper_pos[0] + action[0]*0.07 >= self.x_range[1] or gripper_pos[0] + action[0]*0.07 <= self.x_range[0]:
            action[0] = 0
        if gripper_pos[1] + action[1]*0.07 >= self.y_range[1] or gripper_pos[1] + action[1]*0.07 <= self.y_range[0]:
            action[1] = 0
        return action

    def pusher_tf_update(self, pos):
        offset = np.array([-0.0, 0, 0])
        self.gripper_pos = np.array(pos.data) + offset

    def initialize_goal(self):
        raw_image = None
        while not raw_image:
            raw_image = rospy.wait_for_message("/camera/color/image_raw", Image)
            print('waiting for goal image')
        goal_image = ros_numpy.numpify(raw_image)
        goal_latent, goal_mask = self.get_visual_latent(goal_image)
        return goal_latent, goal_mask

    def get_visual_latent(self, image):
        # TODO crop to square
        crop = image[0:470, 110:580, :]
        image_crop = crop.copy()
        # embed();
        # exit()
        image_crop = cv2.resize(image_crop, (64, 64))
        image_crop_hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)

        light_red = (0, 150, 0)
        dark_red = (20, 255, 255)
        mask = cv2.inRange(image_crop_hsv, light_red, dark_red)
        # cv2.imwrite(os.path.join(IMAGE_PATH, "step{}.png".format(self.step_)), mask)
        # cv2.imshow(image)
        # cv2.waitKey(0)
        # embed();exit()

        tensor = transforms.ToTensor()(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, z, mu, _ = self.vae(tensor)
        latent = mu[0].cpu().numpy()
        # print('current: ',latent)
        # print('goal: ', self.goal_latent)
        image_recon_pil = transforms.ToPILImage()(recon.squeeze().cpu())
        # embed();exit()

        self.mask_publisher.publish(ros_numpy.msgify(Image,np.array(cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)),'mono8'))
        self.image_recon_publisher.publish(ros_numpy.msgify(Image,np.array(cv2.rotate(np.array(image_recon_pil), cv2.cv2.ROTATE_90_CLOCKWISE)),'mono8'))
        self.image_crop_publisher.publish(ros_numpy.msgify(Image,np.array(cv2.rotate(crop, cv2.cv2.ROTATE_90_CLOCKWISE)),'rgb8'))
        self.image_crop_segmentation_publisher.publish(ros_numpy.msgify(Image,np.array(cv2.rotate(image_crop_hsv,cv2.cv2.ROTATE_90_CLOCKWISE)),'rgb8'))
        # if self.step_ % 10 == 0:
        #     cv2.imwrite(os.path.join(IMAGE_PATH, "step{}.png".format(self.step_)), mask)
        #     cv2.imwrite(os.path.join(IMAGE_PATH, "step{}_rec.png".format(self.step_)), np.array(image_recon_pil))

            # cv2.imwrite(os.path.join(IMAGE_PATH, "goal{}_rec.png".format(self.step_)), np.array(goal_recon_pil))
        # embed();exit()
        if self.step_ >= 1:
            with torch.no_grad():
                goal_recon_tensor = self.vae.decode(torch.tensor(self.goal_latent).to(dtype=torch.float32).to(self.device).unsqueeze(0))
                goal_recon_pil = transforms.ToPILImage()(goal_recon_tensor.squeeze().cpu())
                self.goal_publisher.publish(ros_numpy.msgify(Image,cv2.rotate(np.array(goal_recon_pil),cv2.cv2.ROTATE_90_CLOCKWISE) ,'mono8'))

            # publish overall image here
            green_mask = self.get_green_mask(size = 64, mask = mask)
            red_goal = self.get_red_mask(size = 64, mask = self.goal_mask)
            gripper_cross = self.get_gripper_cross(size = 64)

            goal_object_overlay = cv2.addWeighted(green_mask, 0.5, red_goal, 0.5, 0.0)
            goal_object_overlay = cv2.addWeighted(goal_object_overlay, 0.5, gripper_cross, 0.5, 0.0)

            self.overall_publisher.publish(ros_numpy.msgify(Image,np.array(cv2.rotate(goal_object_overlay,cv2.cv2.ROTATE_90_CLOCKWISE)) ,'rgb8'))


            assert self.goal_mask is not None


        return latent, mask

    def get_red_mask(self, size, mask):
        full = np.zeros((size, size, 3), np.uint8)
        full[:] = (255, 0, 0)
        full_red = cv2.bitwise_and(full, full, mask = mask)
        return full_red

    def get_gripper_cross(self, size):
        L = 0.63 # image length in real
        x_start = 0.3
        y_start = 0.44
        full = np.zeros((size, size, 3), np.uint8)
        full[:] = (0,0,255)
        cross_x = int((self.gripper_pos[0] - x_start)/L*size)
        cross_y = int((self.gripper_pos[1] - y_start)/L*size)
        cross_posx = min(max(cross_y, 0), 63)
        cross_posy = min(max(0, cross_x), 63)
        mask_cross = np.zeros((size, size), np.uint8)
        mask_cross[cross_posx, :] = 255
        mask_cross[:, cross_posy] = 255

        blue_cross = cv2.bitwise_and(full, full, mask = mask_cross)
        return blue_cross


    def get_green_mask(self, size, mask):
        full = np.zeros((size, size, 3), np.uint8)
        full[:] = (0, 255, 0)
        full_green = cv2.bitwise_and(full, full, mask = mask)
        return full_green


    def get_obs_dict(self):
        obs = dict()
        # image
        raw_image = None
        while not raw_image:
            raw_image = rospy.wait_for_message("/camera/color/image_raw", Image)
        object_image = ros_numpy.numpify(raw_image)

        object_latent, _ = self.get_visual_latent(object_image)
        goal_latent = self.goal_latent
        # gripper pos
        obs["gripper_pos"] = self.gripper_pos.copy()
        # print('gripper pos', self.gripper_pos.copy())
        obs["observation"] = np.concatenate([object_latent.copy(), obs["gripper_pos"].squeeze().copy()])
        obs["achieved_goal"] = np.concatenate([object_latent.copy(), ])
        obs["desired_goal"] = np.concatenate([goal_latent.copy()])

        return obs

    def step(self, action):
        gripper_pos = self.gripper_pos.copy()
        # clipped_action = self.clip_action(gripper_pos, action)
        clipped_action = action
        # send clipped action to gripper
        action_x = VELOCITY_SCALE * clipped_action[0]
        action_y = VELOCITY_SCALE * clipped_action[1]
        vel_msg = geometry_msgs.msg.Vector3()
        vel_msg.x = action_x
        vel_msg.y = action_y

        self.vel_pub.publish(vel_msg)

        obs = self.get_obs_dict()
        self.step_ += 1
        return obs


def main():
    rospy.init_node('visual_pushing')
    env = VisualPushing()
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

    agent.load(WEIGHT_PATH, args.step, map_location='cuda:1')

    # initialization env
    obs_dict = env.get_obs_dict()
    observation = obs_dict['observation'].copy()
    achieved_goal = obs_dict['achieved_goal'].copy()
    desired_goal = obs_dict['desired_goal'].copy()

    with torch.no_grad():
        for step in range(100000):
            print(f"step: {step}")
            action = agent._select_action(observation, achieved_goal,
                                          desired_goal)  # action is squashed to [-1, 1] by tanh function
            obs_dict_new = env.step(ACTION_SCALE * action)
            observation_new = obs_dict_new['observation'].copy()
            achieved_goal_new = obs_dict_new['achieved_goal'].copy()
            # update stats
            observation[:] = observation_new.copy()
            achieved_goal[:] = achieved_goal_new.copy()
            # print('action: ', action)





if __name__ == '__main__':
    main()



