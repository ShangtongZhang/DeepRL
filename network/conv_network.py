#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .base_network import *

# Network for pixel Atari game with value based methods
class NatureConvNet(nn.Module, VanillaNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(NatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)
        BasicNet.__init__(self, None, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return self.fc5(y)

# Network for pixel Atari game with dueling architecture
class DuelingNatureConvNet(nn.Module, DuelingNet):
    def __init__(self, in_channels, n_actions, optimizer_fn=None, gpu=True):
        super(DuelingNatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_advantage = nn.Linear(512, n_actions)
        self.fc_value = nn.Linear(512, 1)
        BasicNet.__init__(self, None, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        phi = F.relu(self.fc4(y))
        return phi


# Network for pixel Atari game with actor critic
class ActorCriticNatureConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 xentropy_weight=0.01,
                 grad_threshold=40,
                 gpu=True):
        super(ActorCriticNatureConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_actor = nn.Linear(512, n_actions)
        self.fc_critic = nn.Linear(512, 1)
        self.xentropy_weight = xentropy_weight
        self.grad_threshold = grad_threshold
        BasicNet.__init__(self, optimizer_fn=None, gpu=gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = y.view(y.size(0), -1)
        return F.elu(self.fc4(y))

class OpenAIActorCriticConvNet(nn.Module, ActorCriticNet):
    def __init__(self,
                 in_channels,
                 n_actions,
                 LSTM=False):
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
        BasicNet.__init__(self, optimizer_fn=None, gpu=False, LSTM=LSTM)
        if LSTM:
            self.h = self.to_torch_variable(np.zeros((1, hidden_units)))
            self.c = self.to_torch_variable(np.zeros((1, hidden_units)))

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
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
                 n_actions):
        super(OpenAIConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        hidden_units = 256
        self.layer5 = nn.Linear(32 * 3 * 3, hidden_units)
        self.fc6 = nn.Linear(hidden_units, n_actions)

        BasicNet.__init__(self, optimizer_fn=None, gpu=False, LSTM=False)

    def forward(self, x, update_LSTM=True):
        x = self.to_torch_variable(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.view(y.size(0), -1)
        phi = F.elu(self.layer5(y))
        return self.fc6(phi)