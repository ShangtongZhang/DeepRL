#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

# Network for pixel Atari game with value based methods
class NatureConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, n_actions, gpu=0):
        super(NatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return self.fc5(y)

# Network for pixel Atari game with dueling architecture
class DuelingNatureConvNet(nn.Module, DuelingNet):
    def __init__(self, in_channels, n_actions, gpu=0):
        super(DuelingNatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_advantage = nn.Linear(512, n_actions)
        self.fc_value = nn.Linear(512, 1)
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        phi = F.relu(self.fc4(y))
        return phi

class OpenAIActorCriticConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 LSTM=False,
                 gpu=-1):
        super(OpenAIActorCriticConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.LSTM = LSTM
        hidden_units = 256

        if LSTM:
            self.layer5 = nn.LSTMCell(32 * 3 * 3, hidden_units)
        else:
            self.layer5 = nn.Linear(32 * 3 * 3, hidden_units)

        self.fc_actor = nn.Linear(hidden_units, n_actions)
        self.fc_critic = nn.Linear(hidden_units, 1)
        BasicNet.__init__(self, gpu=gpu, LSTM=LSTM)
        if LSTM:
            self.h = self.variable(np.zeros((1, hidden_units)))
            self.c = self.variable(np.zeros((1, hidden_units)))

    def forward(self, x, update_LSTM=True):
        x = self.variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.view(y.size(0), -1)
        if self.LSTM:
            h, c = self.layer5(y, (self.h, self.c))
            if update_LSTM:
                self.h = h
                self.c = c
            phi = h
        else:
            phi = F.elu(self.layer5(y))
        return phi

class OpenAIConvNet(nn.Module, VanillaNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 gpu=0):
        super(OpenAIConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        hidden_units = 256
        self.layer5 = nn.Linear(32 * 3 * 3, hidden_units)
        self.fc6 = nn.Linear(hidden_units, n_actions)

        BasicNet.__init__(self, gpu=gpu, LSTM=False)

    def forward(self, x, update_LSTM=True):
        x = self.variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.view(y.size(0), -1)
        phi = F.elu(self.layer5(y))
        return self.fc6(phi)

class NatureActorCriticConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 gpu=-1):
        super(NatureActorCriticConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 32, 512)

        self.fc_actor = nn.Linear(512, n_actions)
        self.fc_critic = nn.Linear(512, 1)
        BasicNet.__init__(self, gpu=gpu)

    def forward(self, x, _):
        x = self.variable(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        phi = F.relu(self.fc4(x))
        return phi

class CategoricalConvNet(nn.Module, CategoricalNet):
    def __init__(self, in_channels, n_actions, n_atoms, gpu=0):
        super(CategoricalConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_categorical = nn.Linear(512, n_actions * n_atoms)
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        BasicNet.__init__(self, gpu)

    def forward(self, x):
        x = self.variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y