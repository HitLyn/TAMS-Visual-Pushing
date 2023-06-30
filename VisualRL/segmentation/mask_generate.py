import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from IPython import embed

IMAGE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/all_objects_random"
SAVE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/all_objects_masks_random"

for i in range(40000):
    file_name = os.path.join(IMAGE_PATH, "{:0>5d}.png".format(i))
    # image = cv2.resize(cv2.imread(file_name)[200:, 100:400, :], (64, 64)) # with crop
    # embed();exit()
    image = cv2.resize(cv2.imread(file_name), (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    light_red = (0, 150, 0)
    bright_red = (20, 255, 255)
    mask = cv2.inRange(image, light_red, bright_red)

    save_name = os.path.join(SAVE_PATH, "{:0>5d}.png".format(i))
    # embed();exit()
    cv2.imwrite(save_name, mask)



