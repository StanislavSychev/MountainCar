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


class Critic(nn.Module):

    def __init__(self, state_size, hidden_state, hidden_action):
        super(Critic, self).__init__()
        self.ls = nn.Linear(state_size, hidden_state)
        self.la = nn.Linear(1, hidden_action)
        hidden = hidden_action + hidden_state
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, s, a):
        s = self.act(self.ls(s))
        a = self.act(self.la(a))
        x = torch.cat((s, a), 1)
        x = self.act(self.l2(x))
        return self.l3(x)


class Actor(nn.Module):

    def __init__(self, state_size, hidden_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)
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

    def __init__(self, state_dim=2, hidden_actor=100,
                 hidden_action=30, hidden_state=70,
                 memory_size=10000, batch_size=50,
                 eps=0.1, gamma=0.99,
                 alr=3e-4, clr=3e-3, tau=0.01):
        self.actor = Actor(state_dim, hidden_actor)
        self.critic = Critic(state_dim, hidden_action, hidden_state)
        self.t_actor = copy.deepcopy(self.actor)
        self.t_critic = copy.deepcopy(self.critic)
        self.memory = Memory(memory_size, batch_size)
        self.gamma = gamma
        self.eps = eps
        self.a_optim = optim.Adam(self.actor.parameters(), lr=alr)
        self.c_optim = optim.Adam(self.critic.parameters(), lr=clr)
        self.tau = tau
        self.loss = nn.MSELoss()

    def act_random(self, state):
        return np.array([random.uniform(-1, 1)])

    def act_best(self, state):
        a = self.actor(torch.tensor(state).float())
        return np.array([a.item()])

    def act(self, state):
        if random.random() < self.eps:
            return self.act_random(state)
        return self.act_best(state)

    def memorize(self, st1, act, st2, rew, d):
        st1 = torch.tensor(st1).float()
        act = torch.tensor(act).float()
        st2 = torch.tensor(st2).float()
        rew = torch.tensor([rew]).float()
        d = torch.tensor([not d])
        self.memory.add(st1, act, st2, rew, d)

    def update(self):
        # self.t_actor = copy.deepcopy(self.actor)
        # self.t_critic = copy.deepcopy(self.critic)
        self.soft_update(self.critic, self.t_critic, self.tau)
        self.soft_update(self.actor, self.t_actor, self.tau)

    @staticmethod
    def soft_update(current, target, tau):
        for curr_layer, target_layer in zip(current.parameters(), target.parameters()):
            target_layer.data.copy_(tau * curr_layer + (1 - tau) * target_layer)

    def set_eps(self, eps):
        self.eps = eps

    def try_learn(self):
        if not self.memory.ready():
            return
        st, a, stp, r, d = self.memory.sample()
        tq = torch.zeros(d.size()).float()
        with torch.no_grad():
            ap = self.t_actor(stp)
            tq[d] = (self.gamma * self.t_critic(stp, ap.detach()))[d]
            # tq = self.gamma * self.t_critic(stp, ap.detach())
            tq = tq + r
        c_loss = self.loss(self.critic(st, a), tq)
        # update critic
        self.c_optim.zero_grad()
        c_loss.backward()
        self.c_optim.step()
        a_loss = - self.critic(st, self.actor(st)).mean()
        # update actor
        self.a_optim.zero_grad()
        self.c_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

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
        fig.colorbar(c, ax=ax)
        fig.show()

    @staticmethod
    def save(m, name):
        with open('{}.pkl'.format(name), 'wb') as output:
            pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)
        torch.save(m.actor, "{}_actor.pt".format(name))
        torch.save(m.critic, "{}_critic.pt".format(name))

    @staticmethod
    def load(name):
        with open('{}.pkl'.format(name), 'rb') as input:
            m = pickle.load(input)
        m.actor = torch.load("{}_actor.pt".format(name))
        m.critic = torch.load("{}_critic.pt".format(name))
        m.actor.eval()
        m.critic.eval()
        return m


def score(name="mccv0"):
    env = gym.make('MountainCarContinuous-v0')
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
    env = gym.make('MountainCarContinuous-v0')
    m = Policy(gamma=gamma)
    # m = Policy.load("mccv0")
    m.set_eps(0.5)
    max_epoch = 100
    eps_max = 0.2
    eps_min = 0.1
    rr = []
    max_steps = 1500
    won = 0
    for i in range(max_epoch):
        r = 0
        eps = eps_max - i * (eps_max - eps_min) / (max_epoch - 1)
        m.set_eps(eps)
        done = False
        state = env.reset()
        for j in range(max_steps):
            if i == max_epoch - 1:
                env.render()
            action = m.act(state)
            new_state, reward, done, _ = env.step(action)
            rew = reward + 300 * (gamma * abs(new_state[1]) - abs(state[1]))
            # rew = reward
            if done and reward > 0:
                won += 1
            m.memorize(state, action, new_state, rew, done)
            state = new_state
            r += reward
            m.try_learn()
            if done:
                break
        m.update()
        print("{}, {}: {}".format(i, won, r))
        rr.append(r)
    m.visualize()
    Policy.save(m, "mccv0_2")


if __name__ == '__main__':
    train()
    print(score())
