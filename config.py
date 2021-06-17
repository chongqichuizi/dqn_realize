import torch

# 用torch.cuda.current_device()查看当前使用的gpu序列号
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# 环境(游戏名字)
RAM_ENV_NAME = 'LunarLander-v2'  # 用ram作为输入的游戏名
VISUAL_ENV_NAME_0 = 'Pong-v0'   # 用画面输入的游戏名
VISUAL_ENV_NAME_1 = 'Boxing-v0'
CONSTANT_PONG = 90       # padding时候用(画面背景RGB用灰度表示以后是90)
CONSTANT_BOXING = 110.66666667

# agent参数
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

# 训练参数
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 30
EPS_INIT = 1
EPS_DECAY = 0.9
EPS_MIN = 0.05
MAX_T = 1500
NUM_FRAME = 2