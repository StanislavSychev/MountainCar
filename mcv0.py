import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
from torch.autograd import Variable


class DQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l3(x)


class Memory:

    def __init__(self, size, sample_size):
        self.p = 0
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.size = size
        self.sample_size = sample_size

    def add(self, a, b, c, d, e):
        if len(self.a) < self.size:
            self.a.append(a)
            self.b.append(b)
            self.c.append(c)
            self.d.append(d)
            self.e.append(e)
        else:
            self.a[self.p] = a
            self.b[self.p] = b
            self.c[self.p] = c
            self.d[self.p] = d
            self.e[self.p] = e
            self.p = (self.p + 1) % self.size

    def sample(self):
        ind = np.random.choice(len(self.a), self.sample_size, replace=False)
        a = [self.a[i] for i in ind]
        b = [self.b[i] for i in ind]
        c = [self.c[i] for i in ind]
        d = [self.d[i] for i in ind]
        e = [self.e[i] for i in ind]
        return torch.cat(a).view(len(ind), -1), \
               torch.cat(b).view(len(ind), -1), \
               torch.cat(c).view(len(ind), -1), \
               torch.cat(d).view(len(ind), -1), \
               torch.cat(e).view(len(ind), -1)

    def __len__(self):
        return len(self.a)

    def ready(self):
        return len(self.a) >= self.sample_size


class Policy:

    def __init__(self, state_dim=2, action_dim=3, hidden_dim=30,
                 memory_size=5000, batch_size=50,
                 eps=0.1, gamma=0.99, lr=5e-4):
        self.dqn = DQN(state_dim, hidden_dim, action_dim)
        self.target = copy.deepcopy(self.dqn)
        self.memory = Memory(memory_size, batch_size)
        self.gamma = gamma
        self.eps = eps
        self.optim = optim.Adam(self.dqn.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def act_best(self, state):
        q = self.dqn(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
        return torch.max(q, -1)[1].item()

    def act(self, state):
        if random.random() < self.eps:
            return np.random.choice(3, 1)[0]
        return self.act_best(state)

    def memorize(self, st1, act, st2, rew, d):
        st1 = Variable(torch.from_numpy(st1).type(torch.FloatTensor))
        act = torch.tensor([act])
        st2 = Variable(torch.from_numpy(st2).type(torch.FloatTensor))
        rew = torch.tensor([rew]).float()
        d = torch.tensor([d])
        self.memory.add(st1, act, st2, rew, d)

    def update(self):
        self.target = copy.deepcopy(self.dqn)

    def set_eps(self, eps):
        self.eps = eps

    def try_learn(self):
        if not self.memory.ready():
            return
        self.optim.zero_grad()
        st, a, stp, r, d = self.memory.sample()
        tq = torch.zeros(d.size()).float()
        with torch.no_grad():
            tq[d] = self.target(stp).max(1)[0].unsqueeze(1).detach()[d]
        tq = r + tq * self.gamma
        loss = self.loss(self.dqn(st).gather(1, a), tq)
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

    def visualize(self):
        pp = np.linspace(-1.2, 0.6, 101)
        vv = np.linspace(-0.07, 0.07, 101)
        a = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                a[i, j] = self.act_best(np.array([(pp[i] + pp[i + 1]) / 2, (vv[j] + vv[j + 1]) / 2]))
        fig, ax = plt.subplots()
        c = ax.pcolormesh(vv, pp, a)
        ax.set_ylabel('position')
        ax.set_xlabel('velocity')
        fig.colorbar(c, ax=ax, ticks=[0, 1, 2])
        fig.show()

    @staticmethod
    def save(m, name):
        with open('{}.pkl'.format(name), 'wb') as output:
            pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)
        torch.save(m.dqn, "{}_dqn.pt".format(name))

    @staticmethod
    def load(name):
        with open('{}.pkl'.format(name), 'rb') as input:
            m = pickle.load(input)
        m.dqn = torch.load("{}_dqn.pt".format(name))
        m.dqn.eval()
        return m


def score(name="mcv0"):
    env = gym.make('MountainCar-v0')
    p = Policy.load(name)
    res = 0
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action = p.act_best(state)
            state, reward, done, _ = env.step(action)
            res += reward
    return res / 100


def train():
    gamma = 0.99
    render = False
    env = gym.make('MountainCar-v0')
    m = Policy(gamma=gamma, hidden_dim=50)
    max_epoch = 501
    start = 300
    eps_max = 0.1
    eps_min = 0.01
    rr = []
    for i in range(max_epoch):
        r = 0
        if i > start:
            eps = eps_max - (i - start) * (eps_max - eps_min) / (max_epoch - 1 - start)
            m.set_eps(eps)
        done = False
        state = env.reset()
        while not done:
            # if i % 1000 == 0:
            #     env.render()
            action = m.act(state)
            new_state, reward, done, _ = env.step(action)
            rew = reward + 500 * (gamma * abs(new_state[1]) - abs(state[1]))
            m.memorize(state, action, new_state, rew, done)
            state = new_state
            r += reward
            m.try_learn()
        m.update()
        print("{}: {}".format(i, r))
        rr.append(r)
    m.visualize()
    Policy.save(m, "mcv0")


if __name__ == '__main__':
    # train()
    print(score())
