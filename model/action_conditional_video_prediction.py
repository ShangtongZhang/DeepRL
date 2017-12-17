#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchvision
from skimage import io
from collections import deque
import gym
import torch.optim

class Network(nn.Module):
    def __init__(self, num_actions, gpu=True):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(12, 64, 8, 2, (0, 1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, 6, 2, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, 2, (0, 0))

        self.hidden_units = 128 * 11 * 8

        self.fc5 = nn.Linear(self.hidden_units, 2048)
        self.fc6 = nn.Linear(2048, 2048)
        self.fc_action = nn.Linear(num_actions, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, self.hidden_units)

        self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2)
        self.deconv10 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))
        self.deconv11 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))
        self.deconv12 = nn.ConvTranspose2d(128, 3, 8, 2, (0, 1))

        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            self.cuda()
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x.cuda()
        return Variable(x)

    def forward(self, obs, action):
        x = self.to_torch_variable(obs)
        action = self.to_torch_variable(action)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc7(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 11, 8))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = self.deconv12(x)
        return x

def load_episode(game, ep, num_actions):
    path = 'dataset/%s/%05d' % (game, ep)
    with open('%s/action.bin' % (path), 'rb') as f:
        actions = pickle.load(f)
    num_frames = len(actions) + 1
    frames = []
    mean_frame = 0.0

    for i in range(1, num_frames):
        frame = io.imread('%s/%05d.png' % (path, i))
        frame = np.transpose(frame, (2, 0, 1))
        mean_frame += frame
        frames.append(frame)

    mean_frame /= num_frames - 1
    frames = [(frame - mean_frame) / 255.0 for frame in frames]

    actions = actions[1:]
    encoded_actions = np.zeros((len(actions), num_actions))
    encoded_actions[np.arange(len(actions)), actions] = 1

    return frames, encoded_actions

def train(game):
    env = gym.make(game)
    num_actions = env.action_space.n

    net = Network(num_actions)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), 0.001)

    with open('dataset/%s/meta.bin' % (game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    train_episodes = int(episodes * 0.8)
    indices_train = np.arange(train_episodes)
    while True:
        np.random.shuffle(indices_train)
        for ep in indices_train:
            frames, actions = load_episode(game, ep, num_actions)

            buffer = deque(maxlen=4)
            extended_frames = []
            targets = []

            for i in range(len(frames) - 1):
                buffer.append(frames[i])
                if len(buffer) >= 4:
                    extended_frames.append(np.vstack(buffer))
                    targets.append(frames[i + 1])
            actions = actions[3:, :]

            batch_size = 4
            batch_start = 0
            batch_end = batch_start + batch_size
            while batch_start < len(extended_frames):
                x = np.asarray(np.stack(extended_frames[batch_start: batch_end]))
                a = actions[batch_start: batch_end]
                y = np.asarray(np.stack(targets[batch_start: batch_end]))
                y = net.to_torch_variable(y)
                y_ = net(x, a)
                loss = criterion(y_, y)
                print loss.cpu().data.numpy()
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_start = batch_end
                batch_end = min(batch_start + batch_size, len(extended_frames))
