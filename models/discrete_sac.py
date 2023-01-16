from abc import ABC
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from utils import conv2d_output_size
import torch


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class QValueNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(QValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        w, h, c = self.state_shape

        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
        )

        conv1_out = conv2d_output_size((c, h, w), 32, 0, 8, 4)
        conv2_out = conv2d_output_size(conv1_out, 64, 0, 4, 2)
        conv3_out = conv2d_output_size(conv2_out, 64, 0, 3, 1)

        self.fc = nn.Linear(
            in_features=torch.tensor(conv3_out).prod(), out_features=512
        )
        self.q_value = nn.Linear(in_features=512, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, states, **kwargs):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states)
        x = states / 255.0
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        w, h, c = self.state_shape

        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
        )

        conv1_out = conv2d_output_size((c, h, w), 32, 0, 8, 4)
        conv2_out = conv2d_output_size(conv1_out, 64, 0, 4, 2)
        conv3_out = conv2d_output_size(conv2_out, 64, 0, 3, 1)

        self.fc = nn.Linear(
            in_features=torch.tensor(conv3_out).prod(), out_features=512
        )
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, states, **kwargs):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states)
        x = states / 255.0
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return Categorical(probs), probs + z
