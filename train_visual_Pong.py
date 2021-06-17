import gym
import numpy as np
import torch
import os
from agent import *
from config import *
from prepossing import *
CONSTANT = CONSTANT_PONG

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t, num_frame=2, constant=0):       # frame
    # 记录reward和平均reward
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, num_episode+1):
        episodic_reward = 0
        done = False
        frame = env.reset()     # 初始化
        frame = preprocess_pong(frame, constant)        # 预处理
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        t = 0

        while not done and t < max_t:
            t += 1
            action = agent.act(state, eps)      # 根据eps决定行为
            frame, reward, done, _ = env.step(action)      # 进行动作后获取返回值
            frame = preprocess_pong(frame, constant)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            agent.memory.append((state, action, reward, next_state, done))

            if t % 5 == 0 and len(agent.memory) >= agent.bs:
                # 取batch_size训练
                agent.learn()
                agent.soft_update(agent.tau)      # 更新Q^使之别走太远

            state = next_state.copy()       # 更新状态
            episodic_reward += reward       # 获取一个episode累计reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print("\r Episode {}, reward {:.3f}, average reward {:.3f}".format(i, episodic_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()

            eps = max(eps * eps_decay, eps_min)

    return rewards_log


if __name__ == '__main__':
    path = "{}_results".format(VISUAL_ENV_NAME_0)+"\\"
    mkdir(path)
    env = gym.make(VISUAL_ENV_NAME_0)
    agent = Agent(NUM_FRAME, env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, True)
    rewards_log = train(env, agent, VISUAL_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T, NUM_FRAME, CONSTANT_PONG)
    np.save(path+"{}_rewards.npy".format(VISUAL_ENV_NAME_0), rewards_log)
    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), path+"{}_weights.pth".format(VISUAL_ENV_NAME_0))