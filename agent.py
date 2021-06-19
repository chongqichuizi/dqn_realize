import random
import torch
import numpy as np
from collections import deque
import torch.optim as optim
from networks import *

class Agent:
    def __init__(self,state_size, action_size, bs, lr, tau, gamma, device, visual=False):
        '''
        :param state_size: env.observation_space.shape[0]
        :param action_size: env.action_space.n
        :param bs: batch_size 只在sample时候用过，也可以写在外面或者在具体sample的时候定义
        :param lr: learning_rate
        :param tau: soft_update用，让Q^(参数)更新后为Q和Q^的一个线性组合，而非完全复制，更加平滑，算是一个小创新
        :param gamma: y_i中max Q^的衰减系数(参考伪代码)
        :param device: cpu or gpu，在config里面定义
        :param visual: 传入训练数据是画面还是ram参数
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        # 根据输入性质决定用什么网络
        if visual:
            self.Q_local = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
        else:
            self.Q_local = Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Q_Network(self.state_size, self.action_size).to(device)

        self.soft_update(1)     # 让Q^初始化参数等于Q

        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)     # 对Q创建optimizer
        self.memory = deque(maxlen=100000)      # 用来存储state, action, reward, next_state, done(游戏结束)五元组以供训练

    def act(self, state, eps=0):
        # epsilon策略，只不过玩游戏的时候最好别随意, 所以eps为0，所有决策都是agent能想到的最优解
        if random.random() > eps:
            # 把state转成tensor型式数据放到gpu上去算
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)     # 根据state计算出所有actions的values，不追踪梯度
            # 选取value最大的action返回
            # tmp = torch.argmax(action_values).cpu().data.numpy()
            # print(tmp, type(tmp))
            return torch.argmax(action_values).cpu().data.numpy()     # cuda支持的的数据为tensor，只有转移到cpu才能变成numpy
            # return torch.max(action_values)[1].data.numpy()
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        # 随机选取batch_size大小的数据进行训练
        experiences = random.sample(self.memory, self.bs)
        # 对选取出来的五元组分别拼接成列向量，从ndarray转为tensor存在gpu。(对于ndarray使用不太行，用了这种笨方法)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        done = torch.from_numpy(np.vstack([e[4] for e in experiences])).float().to(self.device)

        q_values = self.Q_local(states)     # 根据states计算q值
        # print("Q的值：", q_values)
        # print(type(q_values),type(actions))
        q_values = torch.gather(input=q_values, dim=-1, index=actions)  # 取出每个action对应的q_value

        with torch.no_grad():       # 对于超前的target Q不需要跟踪梯度
            q_targets = self.Q_target(next_states)
            q_targets, _ = torch.max(input=q_targets, dim=-1, keepdim=True)     # 根据next_states计算Q^的值
            q_targets = rewards + self.gamma * (1-done) * q_targets     # 进而计算y
        # 损失函数(均方差)
        loss = (q_values - q_targets).pow(2).mean()

        self.optimizer.zero_grad()      # 计算前先把上一次算的梯度清空
        loss.backward()     # 梯度反向传播
        self.optimizer.step()       # 单步传播

    def soft_update(self, tau):     # 在目标网络参数（接收者）和当前网络参数（生产者）之间线性变化
        for target, local in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target.data.copy_(tau * local.data + (1.0-tau) * target.data)