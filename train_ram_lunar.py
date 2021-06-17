import gym
import numpy as np
import torch
import os
from agent import *
from config import *
from prepossing import *

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    # 记录reward和平均reward
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, num_episode+1):
        episodic_reward = 0
        done = False
        state = env.reset()     # 初始化
        t = 0

        while not done and t < max_t:
            t += 1
            action = agent.act(state, eps)      # 根据eps决定行为
            next_state, reward, done, _ = env.step(action)      # 进行动作后获取返回值
            agent.memory.append((state, action, reward, next_state, done))

            if t % 4 == 0 and len(agent.memory) >= agent.bs:
                # 取batch_size训练
                agent.learn()
                agent.soft_update(agent.tau)      # 更新Q^使之别走太远

            state = next_state.copy()       # 更新状态
            episodic_reward += reward       # 获取一个episode累计reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print("\r Episode {}, reward {:.3f}, average reward {:.3f}".format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print()

            eps = max(eps * eps_decay, eps_min)

    return rewards_log


if __name__ == '__main__':
    path = "{}_results".format(RAM_ENV_NAME)+"\\"
    mkdir(path)
    env = gym.make(RAM_ENV_NAME)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    np.save(path+"{}_rewards.npy".format(RAM_ENV_NAME), rewards_log)
    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), path+"{}_weights.pth".format(RAM_ENV_NAME))