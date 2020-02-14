from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from utils.core import np_to_pytorch_batch
from trainer.trainer import SACTrainer
import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable


class OptTrainer(SACTrainer):
    def __init__(
            self,
            policy_producer,
            q_producer,
            n_estimators=2,
            action_space=None,
            discount=0.99,
            reward_scale=1.0,
            delta=0.95,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            deterministic=True,
            q_min=0,
            q_max=100,
    ):
        super().__init__(policy_producer,
                         q_producer,
                         action_space=action_space,
                         discount=discount,
                         reward_scale=reward_scale,
                         policy_lr=policy_lr,
                         qf_lr=qf_lr,
                         optimizer_class=optimizer_class,
                         soft_target_tau=soft_target_tau,
                         target_update_period=target_update_period,
                         use_automatic_entropy_tuning=use_automatic_entropy_tuning,
                         target_entropy=target_entropy,
                         deterministic=deterministic)

        quantiles = [i * 1. / (n_estimators - 1) for i in range(n_estimators)]
        for p in range(n_estimators):
            if quantiles[p] == delta:
                self.delta_index = p
                break
            if quantiles[p] > delta:
                self.delta_index = p - 1
                break
        self.q_min = q_min
        self.q_max = q_max
        self.delta = delta
        self.n_estimators = n_estimators
        self.qfs = []
        self.qf_optimizers = []
        self.tfs = []
        initial_values = np.linspace(self.q_min, self.q_max, self.n_estimators)
        for i in range(n_estimators):
            self.qfs.append(q_producer(bias=initial_values[i]))
            self.tfs.append(q_producer(bias=initial_values[i]))
            self.qf_optimizers.append(optimizer_class(
                self.qfs[i].parameters(),
                lr=qf_lr,))

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

        # if self.use_automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha *
        #                    (log_pi +
        #                     self.target_entropy).detach()).mean()
        #     self.alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optimizer.step()
        #     alpha = self.log_alpha.exp()
        # else:
        #     alpha_loss = 0
        #     alpha = 1
        qs = [q(obs, new_obs_actions) for q in self.qfs]

        # q_new_actions = torch.min(
        #     self.qf1(obs, new_obs_actions),
        #     self.qf2(obs, new_obs_actions),
        # )
        delta_index = self.delta_index
        qs = torch.stack(qs, dim=0)
        sorted_qs = torch.sort(qs, dim=0)[0]

        # q_new_actions = torch.min(qs, dim=0)[0]
        upper_bound = sorted_qs[delta_index]

        ##upper_bound (in some way)
        policy_loss = (-upper_bound).mean()

        """
        QF Loss
        """
        q_preds = []
        for i in range(len(self.qfs)):
            q_preds.append(self.qfs[i](obs, actions))
        # q1_pred = self.qf1(obs, actions)
        # q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        qs = torch.stack(q_preds, dim=0)
        sorted_qs = torch.sort(qs, dim=0)[0]
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )
        target_qs = [q(next_obs, new_next_actions) for q in self.tfs]
        target_qs = torch.stack(target_qs, dim=0)
        target_qs_sorted = torch.sort(target_qs, dim=0)[0]
        # target_q_values = torch.min(target_qs, dim=0)[0] - alpha * new_log_pi
        target_q_values = target_qs_sorted
        q_target = self.reward_scale * rewards + \
                   (1. - terminals) * self.discount * target_q_values
        qf_losses = []
        qf_loss = 0
        for i in range(len(self.qfs)):
            q_loss = self.qf_criterion(sorted_qs[i], q_target[i].detach())
            qf_losses.append(q_loss)
            qf_loss += q_loss
            self.qf_optimizers[i].zero_grad()
            q_loss.backward(retain_graph=True)
            self.qf_optimizers[i].step()

        """
        # Update networks
        # """
        # self.qf1_optimizer.zero_grad()
        # qf1_loss.backward()
        # self.qf1_optimizer.step()
        #
        # self.qf2_optimizer.zero_grad()
        # qf2_loss.backward()
        # self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            for i in range(len(self.qfs)):
                ptu.soft_update_from_to(
                    self.qfs[i], self.tfs[i], self.soft_target_tau
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
            policy_loss = (upper_bound).mean()
            for i in range(len(self.qfs)):
                self.eval_statistics['QF' + str(i) + ' Loss'] = np.mean(ptu.get_numpy(qf_losses[i]))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q' + str(i) + 'Predictions',
                    ptu.get_numpy(q_preds[i]),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q' + str(i) + 'Targets',
                    ptu.get_numpy(q_target[i]),
                ))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        self._n_train_steps_total += 1



    @property
    def networks(self) -> Iterable[nn.Module]:
        return [self.policy] + self.qfs + self.tfs


    def get_snapshot(self):
        data =  dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),

            log_alpha=self.log_alpha,
            alpha_optim_state_dict=self.alpha_optimizer.state_dict(),

            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )
        qfs_state_dicts = []
        qfs_optims_state_dicts = []
        target_qfs_state_dicts = []
        for i in range(len(self.qfs)):
            qfs_state_dicts.append(self.qfs[i].state_dict())
            qfs_optims_state_dicts.append(self.qf_optimizers[i].state_dict())
            target_qfs_state_dicts.append(self.tfs[i].state_dict())

        data["qfs_state_dicts"] = qfs_state_dicts
        data["qfs_optims_state_dicts"] = qfs_optims_state_dicts
        data["target_qfs_state_dicts"] = target_qfs_state_dicts
        return data


    def restore_from_snapshot(self, ss):

        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        self.qfs_optimizer = []
        self.qfs = []
        self.tfs = []
        qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        target_qfs_state_dicts = ss['target_qfs_state_dicts']
        for i in range(len(qfs_state_dicts)):

            self.qfs[1].load_state_dict(qfs_state_dicts[i])
            self.qfs_optimizer[i].load_state_dict(qfs_optims_state_dicts[i])
            self.tfs[i].load_state_dict(target_qfs_state_dicts[i])

        log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']

        self.log_alpha.data.copy_(log_alpha)
        self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
