from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import np_to_pytorch_batch

import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from utils.core import torch_ify

class DDPGTrainer(object):
    def __init__(
            self,
            policy_producer,
            q_producer,
            action_space=None,
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            deterministic=True,
            use_target_policy=False,
    ):
        super().__init__()

        """
        The class state which should not mutate
        """

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.deterministic = True
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale
        self.use_target_policy = use_target_policy
        """
        The class mutable state
        """
        self.policy = policy_producer()
        self.qf = q_producer()
        self.target_qf = q_producer()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.qfs = [self.qf]
        self.tfs = [self.target_qf]

        if use_target_policy:
            self.target_policy_network = policy_producer()
            ptu.soft_update_from_to(
                self.policy, self.target_policy_network, 1
            )

    def train(self, np_batch):
        buffer = np_batch.pop('buffer', None)
        batch = np_to_pytorch_batch(np_batch)
        batch['buffer'] = buffer
        self.train_from_torch(batch)

    def predict(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch_ify(obs)
            action = torch_ify(action)
        if len(obs.shape) == 1:
            args = list(torch.unsqueeze(i, dim=0) for i in (obs, action))
        else:
            args = list(i for i in (obs, action))

        Q = self.qf[0](*args)
        return Q



    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )

        q_new_actions = self.qf(obs, new_obs_actions)

        policy_loss = (-q_new_actions).mean()

        """
        QF Loss
        """

        q_pred = self.qf(obs, actions)
        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        if self.use_target_policy:
            # maintain a target policy network that is updated slowly as in ddpg for improved stability
            new_next_actions, _, _, new_log_pi, *_ = self.target_policy_network(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )
        else:
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic,
            )

        target_q_values = self.target_qf(next_obs, new_next_actions)

        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_values

        qf_loss = self.qf_criterion(q_pred, q_target.detach())
        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:

            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )
            if self.use_target_policy:
                ptu.soft_update_from_to(
                    self.target_policy_network, self.target_policy_network, self.soft_target_tau
                )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (-q_new_actions).mean()
            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(q_pred), axis=0).mean()
            self.eval_statistics['Q Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        netwroks = [
            self.policy,
            self.qf,
            self.target_qf,
        ]
        if self.use_target_policy:
            netwroks.append(self.target_policy_network)
        return netwroks

    def get_snapshot(self):
        snapshot = dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),
            qf_state_dict=self.qf.state_dict(),
            qf_optim_state_dict=self.qf_optimizer.state_dict(),
            target_qf_state_dict=self.target_qf.state_dict(),
            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )
        if self.use_target_policy:
            snapshot['target_policy_network'] = self.target_policy_network

        return snapshot

    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        qf_state_dict, qf_optim_state_dict = ss['qf_state_dict']
        target_qf_state_dict = ss['target_qf_state_dict']

        self.qf.load_state_dict(qf_state_dict)
        self.qf_optimizer.load_state_dict(qf_optim_state_dict)
        self.target_qf.load_state_dict(target_qf_state_dict)

        if self.use_target_policy:
            target_policy_network_state_dict = ss['target_policy_network']
            self.target_policy_network.load_state_dict(target_policy_network_state_dict)
        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
