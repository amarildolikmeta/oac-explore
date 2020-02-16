import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from trainer.trainer import SACTrainer
import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from scipy.stats import norm
from utils.core import  torch_ify


class GaussianTrainer(SACTrainer):
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
            std_lr=5e-5,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
            deterministic=True,
            q_min=0,
            q_max=100,
            pac=False,
            ensemble=False,
            n_policies=1
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

        self.q_min = q_min
        self.q_max = q_max
        self.standard_bound = standard_bound = norm.ppf(delta, loc=0, scale=1)
        self.pac = pac

        mean = (q_max + q_min) / 2
        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        self.delta = delta
        self.n_estimators = n_estimators
        self.qfs = []
        self.qf_optimizers = []
        self.tfs = []
        self.q = q_producer(bias=mean)
        self.q_target = q_producer(bias=mean)
        self.q_optimizer = optimizer_class(
                self.q.parameters(),
                lr=qf_lr,)
        if self.pac:
            a = (2 + discount) / (2 * (1 - discount))
            b = a - 1
            c = 1

            self.sigma_b = sigma2_0 = (discount * q_max) / (c * standard_bound) * np.sqrt(a * 1000)
            log_std = -5.
            self.std_2 = q_producer(bias=np.log(sigma2_0), positive=True)
            self.std_2_target = q_producer(bias=np.log(sigma2_0), positive=True)
            self.std_2_optimizer = optimizer_class(
                self.std_2.parameters(),
                lr=std_lr * 0.1, )
        self.std = q_producer(bias=log_std, positive=True)
        self.std_target = q_producer(bias=log_std, positive=True)
        self.std_optimizer = optimizer_class(
            self.std.parameters(),
            lr=std_lr, )
        self.qfs = [self.q, self.std]
        self.tfs = [self.q_target, self.std_target]
        if pac:
            self.qfs.append(self.std_2)
            self.tfs.append(self.std_2_target)
        # self.tfs.append(q_producer(bias=std))
        # for i in range(n_estimators):
        #     self.qfs.append()
        #
        #     self.qf_optimizers.append(optimizer_class(
        #         self.qfs[i].parameters(),
        #         lr=qf_lr,))
        self.ensemble = ensemble
        self.n_policies = n_policies
        if ensemble:
            initial_actions = np.linspace(-1., 1., n_policies)
            self.policy = policy_producer(bias=initial_actions, ensemble=ensemble, n_policies=n_policies,
                                          approximator=self)
            self.policy_optimizers = []
            for i in range(self.n_policies):
                self.policy_optimizers.append(optimizer_class(
                    self.policy.policies[i].parameters(),
                    lr=policy_lr))

    def predict(self, obs, action):
        obs = np.array(obs)
        # action = np.array(action)
        obs = torch_ify(obs)
        action = torch_ify(action)
        qs = self.q(obs, action)
        if self.pac:
            stds = self.std_2(obs, action)
        else:
            stds = self.std(obs, action)

        upper_bound = qs + self.standard_bound * stds
        return upper_bound

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']



        """
        QF Loss
        """
        q_preds = self.q(obs, actions)

        # Make sure policy accounts for squashing
        # functions like tanh correctly!

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )
        target_q = self.q_target(next_obs, new_next_actions)

        # target_q_values = torch.min(target_qs, dim=0)[0] - alpha * new_log_pi
        target_q_values = target_q
        q_target = self.reward_scale * rewards + \
                   (1. - terminals) * self.discount * target_q_values
        q_loss = self.qf_criterion(q_preds, q_target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()
        if self.pac:
            std_preds = self.std(obs, actions)
            std_preds_2 = self.std_2(obs, actions)
            target_stds_2 = self.std_2_target(next_obs, new_next_actions)

            std_target = (1. - terminals) * self.discount * target_stds_2
            std_target_2 = (1. - terminals) * self.discount * self.sigma_b / np.sqrt((self._n_train_steps_total + 1))

            std_loss = self.qf_criterion(std_preds, std_target.detach())
            self.std_optimizer.zero_grad()
            std_loss.backward(retain_graph=True)
            self.std_optimizer.step()
            std_loss_2 = self.qf_criterion(std_preds_2, std_target_2.detach())
            self.std_2_optimizer.zero_grad()
            std_loss_2.backward(retain_graph=True)
            self.std_2_optimizer.step()
        else:
            std_preds = self.std(obs, actions)
            target_stds = self.std_target(next_obs, new_next_actions)
            std_target = (1. - terminals) * self.discount * target_stds

            std_loss = self.qf_criterion(std_preds, std_target.detach())
            self.std_optimizer.zero_grad()
            std_loss.backward(retain_graph=True)
            self.std_optimizer.step()
        """
        Policy and Alpha Loss
        """
        if self.ensemble:
            for i in range(self.n_policies):
                new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy.policies[i](
                    obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
                )

                qs = self.q(obs, new_obs_actions)
                if self.pac:
                    stds = self.std_2(obs, new_obs_actions)
                else:
                    stds = self.std(obs, new_obs_actions)
                upper_bound = qs + self.standard_bound * stds

                ##upper_bound (in some way)
                policy_loss = (-upper_bound).mean()
                optimizer = self.policy_optimizers[i]
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
        else:
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs=obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )

            qs = self.q(obs, new_obs_actions)
            if self.pac:
                stds = self.std_2(obs, new_obs_actions)
            else:
                stds = self.std(obs, new_obs_actions)
            upper_bound = qs + self.standard_bound * stds

            ##upper_bound (in some way)
            policy_loss = (-upper_bound).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.q, self.q_target, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.std, self.std_target, self.soft_target_tau
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

            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(q_preds))
            self.eval_statistics['QF std'] = np.mean(ptu.get_numpy(std_preds))
            if self.pac:
                self.eval_statistics['QF std 2'] = np.mean(ptu.get_numpy(std_preds_2))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(q_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_preds),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Target',
                ptu.get_numpy(q_target),
            ))

            self.eval_statistics['STD Loss'] = np.mean(ptu.get_numpy(std_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q STD Predictions',
                ptu.get_numpy(std_preds),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q STD Target',
                ptu.get_numpy(std_target),
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
        if self.ensemble:
            return self.policy.policies + self.qfs + self.tfs
        else:
            return [self.policy] + self.qfs + self.tfs

    def get_snapshot(self):
        data = dict(
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

        qfs_state_dicts.append(self.q.state_dict())
        qfs_optims_state_dicts.append(self.q_optimizer.state_dict())
        target_qfs_state_dicts.append(self.q_target.state_dict())

        qfs_state_dicts.append(self.std.state_dict())
        qfs_optims_state_dicts.append(self.std_optimizer.state_dict())
        target_qfs_state_dicts.append(self.std_target.state_dict())

        data["qfs_state_dicts"] = qfs_state_dicts
        data["qfs_optims_state_dicts"] = qfs_optims_state_dicts
        data["target_qfs_state_dicts"] = target_qfs_state_dicts
        return data

    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        self.qfs_optimizer = []

        qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        target_qfs_state_dicts = ss['target_qfs_state_dicts']

        self.q.load_state_dict(qfs_state_dicts[0])
        self.q_optimizer.load_state_dict(qfs_optims_state_dicts[0])
        self.q_target.load_state_dict(target_qfs_state_dicts[0])

        self.std.load_state_dict(qfs_state_dicts[1])
        self.std_optimizer.load_state_dict(qfs_optims_state_dicts[1])
        self.std_target.load_state_dict(target_qfs_state_dicts[1])

        log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']

        self.log_alpha.data.copy_(log_alpha)
        self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
