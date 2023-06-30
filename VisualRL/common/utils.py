import torch
import torch.nn as nn
import numpy as np
import time
import random
import os


def get_device(device):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        torch.cuda.set_device(int(device))
        return torch.device("cuda")

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_path_for_results(PATH, image = True, model = True):
    train_time = time.strftime("%m_%d-%H_%M", time.gmtime())
    os.makedirs(os.path.join(PATH, train_time), exist_ok = True)
    save_path = os.path.join(PATH, train_time)
    path = []
    if image:
        image_path = os.path.join(save_path, 'image_results')
        os.makedirs(image_path, exist_ok = True)
        path.append(image_path)
    if model:
        model_path = os.path.join(save_path, 'vae_model')
        os.makedirs(model_path, exist_ok = True)
        path.append(model_path)

    return path
