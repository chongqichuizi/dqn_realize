import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Q_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden=[64, 64]):
        super(Q_Network, self).__init__()
        # 三个传播矩阵
        self.fc1 = nn.Linear(state_size, hidden[0]).to(DEVICE)
        self.fc2 = nn.Linear(hidden[0], hidden[1]).to(DEVICE)
        self.fc3 = nn.Linear(hidden[1], action_size).to(DEVICE)

    def forward(self, state):
        x = state
        # 用relu在层间整流
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)     # 输出层不能整流
        return x


class Visual_Q_Network(nn.Module):
    '''
    网络输入数据是(层数(RGB之类的), 80, 80)格式
    '''
    def __init__(self, num_frame, num_action):
        super(Visual_Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2).to(DEVICE)  # 输出图像大小为16x20x20
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2).to(DEVICE)    # 32x9x9
        self.fc1 = nn.Linear(32*81, 256).to(DEVICE)
        self.fc2 = nn.Linear(256, num_action).to(DEVICE)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32*81)   # 做一个flatten， -1表示batch_size不变
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


