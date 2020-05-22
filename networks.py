"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import utils.pytorch_util as ptu
from utils.core import eval_np
import numpy as np

def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            bias=None,
            positive=False,
            train_bias=True
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        self.positive = positive
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        if bias is None:
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            if isinstance(bias, np.ndarray):
                self.last_fc.bias.data = ptu.from_numpy(bias.astype(np.float32))
            else:
                self.last_fc.bias.data.fill_(bias)
        if not train_bias:
            self.last_fc.bias.requires_grad = False

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if self.positive:
            if isinstance(self.positive, list):
                for i, v in enumerate(self.positive):
                    if v:
                        output[:, i] = torch.exp(output[:, i])
            else:
                output = torch.exp(output)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class MultipleMlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            bias=None,
            positive=False,
            n_components=1,
    ):
        super().__init__()

        self.n_components = n_components

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcss = []
        self.last_fcs = []
        self.positive = positive

        for j in range(n_components):
            fcs = []
            in_size = input_size
            for i, next_size in enumerate(hidden_sizes):
                fc = nn.Linear(in_size, next_size)
                in_size = next_size
                hidden_init(fc.weight)
                fc.bias.data.fill_(b_init_value)
                self.__setattr__("fc%s_%s" % (j, i), fc)
                fcs.append(fc)

            self.fcss.append(fcs)

            last_fc = nn.Linear(in_size, output_size)
            self.__setattr__("last_fc%s" % j, last_fc)
            last_fc.weight.data.uniform_(-init_w, init_w)
            if bias is None:
                last_fc.bias.data.uniform_(-init_w, init_w)
            else:
                last_fc.bias.data.fill_(bias)
            self.last_fcs.append(last_fc)

        #print('network', n_components, self.fcss, self.last_fcs)
        #quit()

    def forward(self, input, return_preactivations=False):
        outputs = []
        preactivations = []
        for j in range(self.n_components):
            h = input
            for i, fc in enumerate(self.fcss[j]):
                h = fc(h)
                h = self.hidden_activation(h)
            preactivation = self.last_fc(h)
            preactivations.append(preactivation)
            output = self.output_activation(preactivation)
            if self.positive:
                output = torch.exp(output)
            outputs.append(output)

        if return_preactivations:
            return torch.stack(outputs), torch.stack(preactivations)
        else:
            return torch.stack(outputs)


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)
