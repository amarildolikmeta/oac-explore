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
from torch.distributions import Distribution, Normal
from policies import TanhGaussianPolicy


class GaussianTrainerTS(SACTrainer):
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
            qf_posterior_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=False,
            target_entropy=None,
            deterministic=False,
            q_min=0,
            q_max=100,
            n_components=1,
            share_layers=False,
            q_posterior_producer=None,
            counts=False
    ):
        #print('before', n_components)
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

        #print('after', n_components)
        self.q_min = q_min
        self.q_max = q_max
        self.standard_bound = standard_bound = norm.ppf(delta, loc=0, scale=1)
        self.share_layers = share_layers
        mean = (q_max + q_min) / 2
        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        self.delta = delta
        self.n_estimators = n_estimators
        self.qfs = []
        self.qf_optimizers = []
        self.tfs = []
        self.counts = counts
        if q_posterior_producer is None:
            q_posterior_producer = q_producer
        if share_layers:
            self.q = q_producer(bias=np.array([mean, log_std]), positive=[False, True])
            self.q_target = q_producer(bias=np.array([mean, log_std]), positive=[False, True])
            self.q_optimizer = optimizer_class(
                    self.q.parameters(),
                    lr=qf_lr,)
            self.qfs = [self.q]
            self.tfs = [self.q_target]
        else:
            self.q = q_producer(bias=mean)
            self.q_target = q_producer(bias=mean)
            self.q_optimizer = optimizer_class(
                self.q.parameters(),
                lr=qf_lr, )
            self.std = q_producer(bias=log_std, positive=True)
            self.std_target = q_producer(bias=log_std, positive=True)
            self.std_optimizer = optimizer_class(
                self.std.parameters(),
                lr=std_lr, )
            self.qfs = [self.q, self.std]
            self.tfs = [self.q_target, self.std_target]

        self.q_posterior = q_posterior_producer()
        self.q_posterior_optimizer = optimizer_class(
            self.q_posterior.parameters(),
            lr=qf_posterior_lr, )
        self.q_posterior_criterion = nn.MSELoss()

        self.n_components = n_components

        self.policy = policy_producer(n_components=n_components)
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        #self.policy = policy_producer()

        self.normal = Normal(0, 1)

        assert not self.deterministic

    @property
    def networks(self) -> Iterable[nn.Module]:
        # return [self.policy] + self.qfs + self.tfs
        return self.policy.policies_list + self.qfs + self.tfs + [self.q_posterior]


    def predict(self, obs, action):
        obs = np.array(obs)
        # action = np.array(action)
        obs = torch_ify(obs)
        action = torch_ify(action)
        qs = self.q(obs, action)
        if self.share_layers:
            stds = qs[:, 1].unsqueeze(-1)
            qs = qs[:, 0].unsqueeze(-1)
        else:
            stds = self.std(obs, action)
        return qs, stds

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        n = len(obs)
        if self.counts:
            counts = batch['counts']
            discount = torch.ones_like(counts)
            discount[counts == 0] = self.discount
        else:
            discount = self.discount
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
        if self.share_layers:
            std_preds = q_preds[:, 1].unsqueeze(-1)
            q_preds = q_preds[:, 0].unsqueeze(-1)
            target_stds = target_q_values[:, 1].unsqueeze(-1)
            target_q_values = target_q_values[:, 0].unsqueeze(-1)
            std_target = (1. - terminals) * discount * target_stds
            q_target = self.reward_scale * rewards + \
                       (1. - terminals) * discount * target_q_values
            loss = 0
            q_loss = self.qf_criterion(q_preds, q_target.detach())
            loss += q_loss
            std_loss = self.qf_criterion(std_preds, std_target.detach())
            loss += std_loss
            self.q_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.q_optimizer.step()
        else:
            # q network loss
            q_target = self.reward_scale * rewards + \
                       (1. - terminals) * discount * target_q_values
            q_loss = self.qf_criterion(q_preds, q_target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            self.q_optimizer.step()
            # std network loss
            std_preds = self.std(obs, actions)
            target_stds = self.std_target(next_obs, new_next_actions)
            std_target = (1. - terminals) * discount * target_stds

            std_loss = self.qf_criterion(std_preds, std_target.detach())
            self.std_optimizer.zero_grad()
            std_loss.backward(retain_graph=True)
            self.std_optimizer.step()

        """
        Posterior update
        """
        n_samples = 10

        q_samples_all = []
        for i in range(n_samples):
            z = self.normal.sample((n, 1))
            q_samples = q_preds + z * std_preds
            q_samples_all.append(q_samples)
        q_samples_all = torch.cat(q_samples_all, dim=0)

        q_posterior_preds = self.q_posterior(obs, actions)
        q_posterior_preds = q_posterior_preds.repeat(n_samples, 1)

        q_posterior_loss = self.q_posterior_criterion(q_posterior_preds, q_samples_all.detach())
        self.q_posterior_optimizer.zero_grad()
        q_posterior_loss.backward()
        self.q_posterior_optimizer.step()

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs=obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )
        #print(new_obs_actions)
        #print(torch.mean(policy_log_std))

        q_p_s = self.q_posterior(obs, new_obs_actions)

        policy_loss = (-q_p_s).mean()
        #policy_loss = (-upper_bound).mean()
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
            if not self.share_layers:
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
            policy_loss = (q_p_s).mean()

            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(q_preds))
            self.eval_statistics['QF std'] = np.mean(ptu.get_numpy(std_preds))
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
        if not self.share_layers:
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
        if not self.share_layers:
            self.std.load_state_dict(qfs_state_dicts[1])
            self.std_optimizer.load_state_dict(qfs_optims_state_dicts[1])
            self.std_target.load_state_dict(target_qfs_state_dicts[1])

        log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']

        self.log_alpha.data.copy_(log_alpha)
        self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)

        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
