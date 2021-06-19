import numpy as np
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from config import *
def preprocess_pong(image, constant):
    image = image[34:194, :, :]     # 160, 160, 3
    image = np.mean(image, axis=2, keepdims=False)  # 160, 160
    image = image[::2, ::2]     # 80, 80
    image = image/256
    image = image - constant/256    # remove background
    return image
def preprocess_boxing(image, constant):
    image = image[36:176, 32:128, :]    # 140, 96, 3
    image = np.mean(image, axis=2, keepdims=False)  # 140, 96
    image = image[::2, ::2]     # 70, 48
    image = image/256
    image = image - constant/256    # remove background
    return image