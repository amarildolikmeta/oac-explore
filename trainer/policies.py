import numpy as np
import torch
from torch import nn as nn

from utils.core import eval_np
from networks import Mlp, MultipleMlp
import torch
from torch.distributions import Distribution, Normal, Categorical
from trainer.mixture_same_family import MixtureSameFamily
import utils.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhMixtureNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, n_components, weights, normal_means, normal_stds, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """

        #print(normal_means.shape, normal_stds.shape)

        self.n_components = n_components
        self.normal_means = normal_means
        self.normal_stds = normal_stds

        self.batch_size = len(self.normal_means)

        self.weights = weights.repeat(self.batch_size, 1)

        self.mix = Categorical(self.weights)
        self.comp = Normal(self.normal_means, self.normal_stds)

        #print(self.mix, self.comp)
        #quit()

        self.mixture = MixtureSameFamily(self.mix, self.comp)

        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.mixture.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        norm_log_probs = self.mixture.log_prob(pre_tanh_value)
        return norm_log_probs - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.mixture.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    @property
    def mean(self):
        return self.mixture.mean

    @property
    def variance(self):
        return self.mixture.variance

    def rsample(self, return_pretanh_value=False):
        """
                Sampling in the reparameterization case.
                """
        mix_sample = Categorical(self.weights).sample()
        mix_sample_r = mix_sample.reshape([mix_sample.shape[0], 1])

        means_ = torch.gather(self.normal_means, 1, mix_sample_r)
        stds_ = torch.gather(self.normal_stds, 1, mix_sample_r)
        z = (
                means_ +
                stds_ *
                Normal(
                    ptu.zeros(means_.size()),
                    ptu.ones(stds_.size())
                ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        #print(normal_mean, normal_std)

        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                ptu.zeros(self.normal_mean.size()),
                ptu.ones(self.normal_std.size())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class TanhGaussianPolicy(Mlp):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            bias=None,
            **kwargs
    ):
        if bias is not None:
            bias = np.arctanh(bias)
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            bias=bias,
            **kwargs
        )
        print('PARAMETERS BEFORE', list(self.parameters()))
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.policies_list = [self]
        print('PARAMETERS AFTER', list(self.parameters()))

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        if log_prob is None:
            log_prob = torch.zeros_like(action)
            pre_tanh_value = mean
        return (
            action, mean, log_std, log_prob, std,
            pre_tanh_value,
        )

    def reset(self):
        pass
'''
class TanhGaussianMixturePolicy(nn.Module):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            bias=None,
            n_components=1,
            **kwargs
    ):
        super().__init__()
        self.n_components = n_components
        self.policies_list = [TanhGaussianPolicy(hidden_sizes,
                                       obs_dim,
                                       action_dim,
                                       std,
                                       init_w,
                                       bias,
                                       **kwargs)
                            for _ in range(n_components)]
        self.policies = nn.ModuleList(self.policies_list)

        self.weights = nn.Parameter(torch.ones(n_components) / n_components, requires_grad=True)

        print('PARAMETERS', list(self.parameters()))

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]



    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        self.categorical = Categorical(self.weights)
        idx = self.categorical.sample_n(1)
        action, mean, log_std, log_prob, std, pre_tanh_value = self.policies[idx].forward(obs,
                                                                                        reparameterize,
                                                                                        deterministic,
                                                                                        return_log_prob)

        log_prob = log_prob + torch.log(self.weights[idx])

        #print(self.weights)

        if log_prob is None:
            log_prob = torch.zeros_like(action)
            pre_tanh_value = torch.zeros_like(action)
        return (
            action, mean, log_std, log_prob, std,
            pre_tanh_value,
        )

    def reset(self):
        pass

    def to(self, **args):
        for i in range(len(self.policies)):
            self.policies[i].to(**args)

'''
class TanhGaussianMixturePolicy(MultipleMlp):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            bias=None,
            n_components=1,
            **kwargs
    ):
        if bias is not None:
            bias = np.arctanh(bias)
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            bias=bias,
            n_components=n_components,
            **kwargs
        )
        # print('PARAMETERS BEFORE', list(self.parameters()))
        self.log_std = None
        self.std = std

        self.last_fc_log_stds = []
        for j in range(n_components):
            if std is None:
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                last_fc_log_std.bias.data.uniform_(-init_w, init_w)
                self.__setattr__("last_fc_log_std%s" % j, last_fc_log_std)
                self.last_fc_log_stds.append(last_fc_log_std)
            else:
                self.log_std = np.log(self.std)
                assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        #self.weights = nn.Parameter(torch.ones(n_components) / n_components, requires_grad=True)
        self.weights = torch.ones(n_components) / n_components

        self.policies_list = [self]
        #print('PARAMETERS AFTER', list(self.parameters()))

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        #print(obs_np)
        #print(actions)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        #print(obs_np)
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_all_components=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        means = []
        stds = []
        log_stds = []

        for j in range(self.n_components):
            h = obs
            for i, fc in enumerate(self.fcss[j]):
                #print(h, fc)
                h = self.hidden_activation(fc(h))
            #print(h, self.last_fcs[j])
            mean = self.last_fcs[j](h)
            #print('%s %s' %(j, mean))
            if self.std is None:
                log_std = self.last_fc_log_stds[j](h)
                log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = torch.exp(log_std)
            else:
                std = self.std
                log_std = self.log_std
            means.append(mean)
            stds.append(std)
            log_stds.append(log_std)

        #print(obs, means)
        means = torch.cat(means, dim=1)
        stds = torch.cat(stds, dim=1)
        log_stds = torch.cat(log_stds, dim=1)

        log_prob = None
        pre_tanh_value = None
        if deterministic:
            raise NotImplementedError #action = torch.tanh(mean)
        else:
            tanh_mixture = TanhMixtureNormal(self.n_components, self.weights, means, stds)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_mixture.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_mixture.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_mixture.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_mixture.rsample()
                else:
                    action = tanh_mixture.sample()
            mean_ = tanh_mixture.mean
            std_ = tanh_mixture.stddev
            log_std_ = torch.log(std_)
        if log_prob is None:
            log_prob = torch.zeros_like(action)
            pre_tanh_value = torch.zeros_like(action)
        #print('FW', action)
        if return_all_components:
            return (
                action, means, log_stds, log_prob, stds,
                pre_tanh_value,
            )
        else:
            return (
                action, mean_, log_std_, log_prob, std_,
                pre_tanh_value,
            )

    def reset(self):
        pass


class MakeDeterministic(object):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, deterministic=True):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, obs_np, deterministic=True):
        return self.stochastic_policy.get_actions(obs_np, deterministic=True)

    def reset(self):
        pass

    def forward(self, obs, reparameterize=True, deterministic=False, return_log_prob=False,):
        return self.stochastic_policy.forward(obs, reparameterize=reparameterize, deterministic=True,
                                              return_log_prob=return_log_prob)
    def load_state_dict(self, **args):
        return self.stochastic_policy.load_state_dict(**args)

    def load_state_dict(self, **args):
        return self.stochastic_policy.load_state_dict(**args)

    def state_dict(self, **args):
        return self.stochastic_policy.state_dict(**args)

    def to(self, **args):
        return self.stochastic_policy.to(**args)

def policy_producer(obs_dim, action_dim, hidden_sizes, deterministic=False, ensemble=False, n_policies=1, n_components=1,
                    approximator=None, bias=None):
    if ensemble:
        policy = EnsemblePolicy(approximator=approximator,
                                hidden_sizes=hidden_sizes,
                                obs_dim=obs_dim,
                                action_dim=action_dim,
                                n_policies=n_policies,
                                bias=bias)
    else:
        if n_components > 1:
            policy = TanhGaussianMixturePolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                n_components=n_components,
            )
        else:
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
            )

        if deterministic:
            policy = MakeDeterministic(policy)

    return policy


class EnsemblePolicy:
    def __init__(
            self,
            approximator,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            bias=None,
            n_policies=1,
            share_layers=False,
            **kwargs
    ):
        self.share_layers = share_layers
        self.policies = []
        self.n_policies = n_policies
        if share_layers:
            action_dim = action_dim * self.n_policies
            self.policies.append(TanhGaussianPolicy(hidden_sizes, obs_dim, action_dim, std, init_w, bias,
                                                    **kwargs))
        else:
            for i in range(n_policies):
                self.policies.append(TanhGaussianPolicy(hidden_sizes, obs_dim, action_dim, std, init_w, bias[i],
                                                    **kwargs))
        self.approximator = approximator

    def __call__(self, **args):
        return self.forward(**args)

    def forward(self,
                obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False,
                ):
        actions = []
        values = []
        results = []
        if self.share_layers:
            results = self.policies[0].forward(obs, reparameterize, deterministic, return_log_prob)
            actions = results[0]
            actions = actions.reshape(actions.shape[0], self.n_policies, -1)
            for i in range(self.n_policies):
                values.append(self.approximator.predict(obs, actions[:, i]))
            results = torch.stack([results[j] for j in range(len(results))], dim=-1)
            results = results.permute(0, 2, 1)
            results = results.reshape(results.shape[0], results.shape[1], self.n_policies, -1)
            results = results.permute(0, 2, 1, -1)
        else:
            for i in range(len(self.policies)):
                res = self.policies[i].forward(obs, reparameterize, deterministic, return_log_prob)
                actions.append(res[0])
                values.append(self.approximator.predict(obs, res[0]))
                results.append(torch.stack([res[j] for j in range(len(res))], dim=0))
            results = torch.stack(results, dim=0).permute([2, 0, 1, -1])
        values = torch.stack(values, dim=0)
        argmax = torch.max(values, dim=0)[1]
        max_res = results[torch.arange(results.shape[0]), argmax[:, 0]].permute([1, 0, -1])

        return tuple([max_res[i] for i in range(max_res.shape[0])])

    def get_action(self, obs_np, deterministic=False):
        actions = []
        values = []
        max_value = -np.inf
        max_res = None
        if self.share_layers:
            action = self.get_actions(obs_np[None], deterministic=deterministic)
            action = action.reshape(action.shape[0], self.n_policies, -1)
            for i in range(self.n_policies):
                values.append(self.approximator.predict([obs_np], action[:,i])[0])
                if values[i] > max_value:
                    max_value = values[i]
                    max_res = action[:,i]
        else:
            for i in range(len(self.policies)):
                action = self.get_actions(obs_np[None], deterministic=deterministic, index=i)
                actions.append(action)
                values.append(self.approximator.predict([obs_np], action)[0])
                if values[i] > max_value:
                    max_value = values[i]
                    max_res = action
        if max_res is None:
            print("What?")
        return max_res[0, :], {}

    def get_actions(self, obs_np, deterministic=False, index=0):
        return eval_np(self.policies[index], obs_np, deterministic=deterministic)[0]

    def reset(self):
        pass

    def load_state_dict(self, state_dicts):
        for i in range(len(self.policies)):
            self.policies[i].load_state_dict(state_dicts[i])

    def state_dict(self, **args):
        state_dict = []
        for i in range(len(self.policies)):
            state_dict.append(self.policies[i].state_dict(**args))
        return state_dict

    def to(self, **args):
        for i in range(len(self.policies)):
            self.policies[i].to(**args)
        #return self.stochastic_policy.to(**args)

class MixturePolicy:
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            bias=None,
            n_policies=1,
            **kwargs
    ):
        self.policies = []
        for i in range(n_policies):
            self.policies.append(TanhGaussianPolicy(hidden_sizes, obs_dim, action_dim, std, init_w, bias,
                                                    **kwargs))

    def __call__(self, **args):
        return self.forward(**args)

    def forward(self,
                obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False,
                ):
        actions = []
        values = []
        results = []



        for i in range(len(self.policies)):
            action, mean, log_std, log_prob, std, pre_tanh_value = self.policies[i].forward(obs, reparameterize, deterministic, return_log_prob)

            actions.append(res[0])
            values.append(self.approximator.predict(obs, res[0]))
            results.append(torch.stack([res[j] for j in range(len(res))], dim=0))
        values = torch.stack(values, dim=0)
        results = torch.stack(results, dim=0).permute([2, 0, 1, -1])
        argmax = torch.max(values, dim=0)[1]
        max_res = results[torch.arange(results.shape[0]), argmax[:, 0]].permute([1, 0, -1])

        return tuple([max_res[i] for i in range(max_res.shape[0])])

    def get_action(self, obs_np, deterministic=False):
        actions = []
        values = []
        max_value = -np.inf
        max_index = -1
        max_res = None
        for i in range(len(self.policies)):
            action = self.get_actions(obs_np[None], deterministic=deterministic, index=i)
            actions.append(action)
            values.append(self.approximator.predict([obs_np], action)[0])
            if values[i] > max_value:
                max_value = values[i]
                max_index = i
                max_res = action
        return max_res[0, :], {}

    def get_actions(self, obs_np, deterministic=False, index=0):
        return eval_np(self.policies[index], obs_np, deterministic=deterministic)[0]

    def reset(self):
        pass

    def load_state_dict(self, state_dicts):
        for i in range(len(self.policies)):
            self.policies[i].load_state_dict(state_dicts[i])

    def state_dict(self, **args):
        state_dict = []
        for i in range(len(self.policies)):
            state_dict.append(self.policies[i].state_dict(**args))
        return state_dict

    def to(self, **args):
        for i in range(len(self.policies)):
            self.policies[i].to(**args)
        #return self.stochastic_policy.to(**args)