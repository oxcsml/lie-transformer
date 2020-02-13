
from torch import nn
import math


class MLP(nn.Module):

    def __init__(self, n_inpt, n_hiddens, activation=None, activate_final=False, weight_init=True,
                 mode='truncated_normal'):
        super(MLP, self).__init__()

        if activation is None:
            activation = nn.ReLU()

        self._activation = activation

        self._layers = []
        for n_hidden in n_hiddens:
            self._layers.append(nn.Linear(n_inpt, n_hidden))
            self._layers.append(activation)
            n_inpt = n_hidden

        if not activate_final:
            self._layers = self._layers[:-1]

        self.net = nn.Sequential(*self._layers)

        if weight_init:
            self.weight_init(mode=mode)

    def weight_init(self, mode='xavier'):
        if mode == 'xavier':
            print('We are now using xavier')
            initializer = xavier_init
        elif mode == 'truncated_normal':
            print('We are now using truncated normals')
            initializer = truncated_normal_init
        else:
            raise NotImplementedError

        for block in self._modules:
            if block == '_activation':
                continue
            else:
                for m in self._modules[block]:
                    initializer(m)

    def forward(self, x):
        return self.net(x)


def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def truncated_normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        truncated_normal(m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[1]))
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def truncated_normal(tensor, mean, std):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
