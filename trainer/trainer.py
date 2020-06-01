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

class SACTrainer(object):
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

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            deterministic=False
    ):
        super().__init__()

        """
        The class state which should not mutate
        """
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = - \
                    np.prod(action_space.shape).item()

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.deterministic = deterministic
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.discount = discount
        self.reward_scale = reward_scale

        """
        The class mutable state
        """
        self.policy = policy_producer()
        # self.qfs = []
        # self.qf_optimizers = []
        # self.tfs = []
        # for i in range(n_estimators):
        #     self.qfs.append(q_producer())
        #     self.tfs.append(q_producer())
        #     self.qf_optimizers.append(optimizer_class(
        #         self.qfs[i].parameters(),
        #         lr=qf_lr,))
        self.qf1 = q_producer()
        self.qf2 = q_producer()
        self.target_qf1 = q_producer()
        self.target_qf2 = q_producer()

        if self.use_automatic_entropy_tuning:
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.qfs = [self.qf1, self.qf2]
        self.tfs = [self.target_qf1, self.target_qf2]

    def train(self, np_batch):
        buffer = np_batch.pop('buffer', None)
        batch = np_to_pytorch_batch(np_batch)
        batch['buffer'] = buffer
        self.train_from_torch(batch)

    def predict(self, obs, action,  upper_bound=True, beta_UB=4.46, both_values=False):
        if isinstance(obs, np.ndarray):
            obs = torch_ify(obs)
            action = torch_ify(action)
        if len(obs.shape) == 1:
            args = list(torch.unsqueeze(i, dim=0) for i in (obs, action))
        else:
            args = list(i for i in (obs, action))
        Q1 = self.qfs[0](*args)
        Q2 = self.qfs[1](*args)
        mu_Q = (Q1 + Q2) / 2.0
        sigma_Q = torch.abs(Q1 - Q2) / 2.0
        if both_values:
            return mu_Q, sigma_Q
        if not upper_bound:
            return mu_Q

        Q_UB = mu_Q + beta_UB * sigma_Q
        return Q_UB


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
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi +
                            self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 0

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        #qs = [q(obs, new_obs_actions) for q in self.qfs]
        # delta = self.get_delta()
        # qs = torch.stack(qs, dim=0)
        # q_new_actions = torch.min(qs, dim=0)[0]
        # ##upper_bound (in some way)
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        # q_preds = []
        # for i in range(len(self.qfs)):
        #     q_preds.append(self.qfs[i](obs, actions))
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic,
        )
        # target_qs = [q(next_obs, new_next_actions) for q in self.tfs]
        # target_qs = torch.stack(target_qs, dim=0)
        # target_q_values = torch.min(target_qs, dim=0)[0] - alpha * new_log_pi
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_values
        # qf_losses = []
        # qf_loss = 0
        # for i in range(len(self.qfs)):
        #     q_loss = self.qf_criterion(q_preds[i], q_target.detach())
        #     qf_losses.append(q_loss)
        #     qf_loss += q_loss
        #     self.qf_optimizers[i].zero_grad()
        #     q_loss.backward()
        #     self.qf_optimizers[i].step()
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            # for i in range(len(self.qfs)):
            #     ptu.soft_update_from_to(
            #     self.qfs[i], self.tfs[i], self.soft_target_tau
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
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
            policy_loss = (log_pi - q_new_actions).mean()
            # for i in range(len(self.qfs)):
            #     self.eval_statistics['QF' + str(i) + ' Loss'] = np.mean(ptu.get_numpy(qf_losses[i]))
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         'Q' + str(i) + 'Predictions',
            #         ptu.get_numpy(q_preds[i]),
            #     ))
            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(torch.stack([q1_pred, q2_pred], dim=0))
                                                      , axis=0).mean()
            self.eval_statistics['QF std'] = np.std(ptu.get_numpy(torch.stack([q1_pred, q2_pred], dim=0))
                                                    , axis=0).mean()
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        # return [self.policy] + self.qfs + self.tfs
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        snapshot = dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),
            qf1_state_dict=self.qf1.state_dict(),
            qf1_optim_state_dict=self.qf1_optimizer.state_dict(),
            target_qf1_state_dict=self.target_qf1.state_dict(),
            qf2_state_dict=self.qf2.state_dict(),
            qf2_optim_state_dict=self.qf2_optimizer.state_dict(),
            target_qf2_state_dict=self.target_qf2.state_dict(),
            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )

        if self.use_automatic_entropy_tuning:
            snapshot['log_alpha'] = self.log_alpha
            snapshot['alpha_optim_state_dict'] = self.alpha_optimizer.state_dict()
        return snapshot

        # qfs_state_dicts = []
        # qfs_optims_state_dicts = []
        # target_qfs_state_dicts = []
        # for i in range(len(self.qfs)):
        #     qfs_state_dicts.append(self.qfs[i].state_dict())
        #     qfs_optims_state_dicts.append(self.qf_optimizers[i].state_dict())
        #     target_qfs_state_dicts.append(self.tfs[i].state_dict())
        #
        # data["qfs_state_dicts"] = qfs_state_dicts
        # data["qfs_optims_state_dicts"] = qfs_optims_state_dicts
        # data["target_qfs_state_dicts"] = target_qfs_state_dicts
        # return data


    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        # self.qfs_optimizer = []
        # self.qfs = []
        # self.tfs = []
        # qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        # target_qfs_state_dicts = ss['target_qfs_state_dicts']
        # for i in range(len(qfs_state_dicts)):
        #
        #     self.qfs[1].load_state_dict(qfs_state_dicts[i])
        #     self.qfs_optimizer[i].load_state_dict(qfs_optims_state_dicts[i])
        #     self.tfs[i].load_state_dict(target_qfs_state_dicts[i])
        qf1_state_dict, qf1_optim_state_dict = ss['qf1_state_dict'], ss['qf1_optim_state_dict']
        target_qf1_state_dict = ss['target_qf1_state_dict']

        self.qf1.load_state_dict(qf1_state_dict)
        self.qf1_optimizer.load_state_dict(qf1_optim_state_dict)
        self.target_qf1.load_state_dict(target_qf1_state_dict)

        qf2_state_dict, qf2_optim_state_dict = ss['qf2_state_dict'], ss['qf2_optim_state_dict']
        target_qf2_state_dict = ss['target_qf2_state_dict']

        self.qf2.load_state_dict(qf2_state_dict)
        self.qf2_optimizer.load_state_dict(qf2_optim_state_dict)
        self.target_qf2.load_state_dict(target_qf2_state_dict)
        if self.use_automatic_entropy_tuning:
            log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']
            self.log_alpha.data.copy_(log_alpha)
            self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
