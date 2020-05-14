import numpy as np
import torch.optim as optim
import torch
from torch import nn as nn
from trainer.trainer import SACTrainer
import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from scipy.stats import norm
from utils.core import torch_ify, optimize_policy
from utils.misc import mellow_max


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
            std_lr=3e-3,
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
            n_policies=1,
            share_layers=False,
            r_mellow_max=1.,
            b_mellow_max=None,
            mellow_max=False,
            counts=False,
            mean_update=False,
            global_opt=False,
            std_soft_update=False,
            std_soft_update_prob=0.,
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

        self.action_space = action_space
        self.q_min = q_min
        self.q_max = q_max
        self.standard_bound = standard_bound = norm.ppf(delta, loc=0, scale=1)
        self.share_layers = share_layers
        mean = (q_max + q_min) / 2
        std = (q_max - q_min) / np.sqrt(12)
        log_std = np.log(std)
        self.delta = delta
        self.std_init = std
        self.n_estimators = n_estimators
        self.qfs = []
        self.qf_optimizers = []
        self.tfs = []
        self.r_mellow_max = r_mellow_max
        self.b = b_mellow_max
        self.mellow_max = mellow_max
        self.counts = counts
        self.mean_update = mean_update
        self.global_opt = global_opt

        self.std_soft_update = std_soft_update
        self.std_soft_update_prob = std_soft_update_prob

        assert not self.counts or not self.std_soft_update

        if share_layers:
            self.q = q_producer(bias=np.array([mean, log_std]), positive=[False, True])
            self.q_target = q_producer(bias=np.array([mean, log_std]), positive=[False, True])
            self.q_optimizer = optimizer_class(
                self.q.parameters(),
                lr=qf_lr, )
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
            if action_space.shape[0] > 1:
                initial_actions = np.random.uniform(low=action_space.low,
                                                    high=action_space.high,
                                                    size=(n_policies, action_space.shape[0])).flatten()
            else:
                initial_actions = np.linspace(-1., 1., n_policies)
                initial_actions[0] += 5e-2
                initial_actions[-1] -= 5e-2
            self.policy = policy_producer(bias=initial_actions, ensemble=ensemble, n_policies=n_policies,
                                          approximator=self, share_layers=share_layers)
            self.policy_optimizers = []
            if share_layers:
                self.policy_optimizers.append(optimizer_class(
                    self.policy.policies[0].parameters(),
                    lr=policy_lr))
            else:
                for i in range(self.n_policies):
                    self.policy_optimizers.append(optimizer_class(
                        self.policy.policies[i].parameters(),
                        lr=policy_lr))
        elif self.global_opt:
            self.init_policy = policy_producer()
        # if mean_update:
        self.target_policy = policy_producer()
        self.target_policy_optimizer = optimizer_class(
            self.target_policy.parameters(),
            lr=policy_lr)
        if self.global_opt:
            self.init_target_policy = policy_producer()

    def predict(self, obs, action, std=True):
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
        upper_bound = qs + self.standard_bound * stds
        if std:
            return [qs, stds], upper_bound
        return upper_bound

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # if self.counts:
        #     counts = batch['counts']
        #     discount = torch.ones_like(counts)
        #     discount[counts == 0] = self.discount
        # else:
        discount = self.discount


        """
        QF Loss
        """
        q_preds = self.q(obs, actions)

        # Make sure policy accounts for squashing
        # functions like tanh correctly!
        if self.mean_update:
            new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )
        else:
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

            if self.std_soft_update:
                current_stds = std_preds.detach()
                next_stds = (1. - terminals) * discount * target_stds
                std_target = self.std_soft_update_prob * next_stds + (1 - self.std_soft_update_prob) * current_stds

            if self.counts:
                counts = batch['counts']
                factor = torch.zeros_like(counts)
                factor[counts == 0] = 1
                std_target = std_target * factor + (1 - factor) * std_preds


            q_target = self.reward_scale * rewards + \
                       (1. - terminals) * self.discount * target_q_values
            std_target = torch.clamp(std_target, 0, self.std_init)
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
                       (1. - terminals) * self.discount * target_q_values
            q_loss = self.qf_criterion(q_preds, q_target.detach())
            self.q_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            self.q_optimizer.step()
            # std network loss

            std_preds = self.std(obs, actions)
            target_stds = self.std_target(next_obs, new_next_actions)
            std_target = (1. - terminals) * discount * target_stds

            if self.std_soft_update:
                current_stds = std_preds.detach()
                next_stds = (1. - terminals) * discount * target_stds
                std_target = self.std_soft_update_prob * next_stds + (1 - self.std_soft_update_prob) * current_stds

            if self.counts:
                counts = batch['counts']
                factor = torch.zeros_like(counts)
                factor[counts == 0] = 1
                std_target = std_target * factor + (1 - factor) * std_preds

            std_target = torch.clamp(std_target, 0, self.std_init)
            std_loss = self.qf_criterion(std_preds, std_target.detach())
            self.std_optimizer.zero_grad()
            std_loss.backward(retain_graph=True)
            self.std_optimizer.step()

        """
        Policy and Alpha Loss
        """
        if self.ensemble:
            if self.share_layers:
                new_obs_actions_all, policy_mean_all, policy_log_std_all, log_pi_all, *_ = self.policy.policies[0](
                    obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
                )
                new_obs_actions_all = new_obs_actions_all.reshape(new_obs_actions_all.shape[0], self.n_policies, -1)
                policy_mean_all = policy_mean_all.reshape(new_obs_actions_all.shape[0], self.n_policies, -1)
                policy_log_std_all = policy_log_std_all.reshape(new_obs_actions_all.shape[0], self.n_policies, -1)
                log_pi_all = log_pi_all.reshape(new_obs_actions_all.shape[0], self.n_policies, -1)
                upper_bounds = []
                for i in range(self.n_policies):
                    new_obs_actions, policy_mean, policy_log_std, log_pi = new_obs_actions_all[:, i], \
                                                                           policy_mean_all[:, i], \
                                                                           policy_log_std_all[:, i], \
                                                                           log_pi_all[:, i]
                    qs = self.q(obs, new_obs_actions)
                    stds = qs[:, 1].unsqueeze(-1)
                    qs = qs[:, 0].unsqueeze(-1)
                    upper_bound = qs + self.standard_bound * stds

                    ##upper_bound (in some way)
                    policy_loss = (upper_bound).mean()
                    upper_bounds.append(policy_loss * self.r_mellow_max)
                    # total_loss += torch.exp(self.r_mellow_max * policy_loss)
                # log exp trick
                if self.mellow_max:
                    mellow_max_loss = mellow_max(upper_bounds, self.r_mellow_max, self.b)
                    # if self.b:
                    #     b = self.b
                    # else:
                    #     b = torch.stack(upper_bounds).max(0)[0]
                    # mellow_max_loss = 0
                    # for i in range(self.n_policies):
                    #     mellow_max_loss += torch.exp(upper_bounds[i] - b)
                    # if self.b:
                    #     mellow_max_loss = torch.log(mellow_max_loss) / self.r_mellow_max
                    # else:
                    #     mellow_max_loss = (torch.log(mellow_max_loss) + b) / self.r_mellow_max
                    policy_loss = -mellow_max_loss
                else:
                    loss = 0
                    for i in range(self.n_policies):
                        loss += upper_bounds[i]
                    policy_loss = - loss
                optimizer = self.policy_optimizers[0]
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()
            else:
                for i in range(self.n_policies):
                    new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy.policies[i](
                        obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
                    )

                    qs = self.q(obs, new_obs_actions)
                    if self.share_layers:
                        stds = qs[:, 1].unsqueeze(-1)
                        qs = qs[:, 0].unsqueeze(-1)
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
            if self.global_opt:
                pass
                # optimize_policy(self.policy, self.policy_optimizer, batch['buffer'],
                #                 action_space=self.action_space, obj_func=self.obj_func,
                #                 init_policy=self.init_policy, upper_bound=True)
            else:

                qs = self.q(obs, new_obs_actions)
                if self.share_layers:
                    stds = qs[:, 1].unsqueeze(-1)
                    qs = qs[:, 0].unsqueeze(-1)
                else:
                    stds = self.std(obs, new_obs_actions)
                upper_bound = qs + self.standard_bound * stds

                ##upper_bound (in some way)
                policy_loss = (-upper_bound).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

        # if self.mean_update:
        """
            Update_target_policy
        """
        # if self.global_opt:
        #     # optimize_policy(self.target_policy, self.target_policy_optimizer, batch['buffer'],
        #     #                 action_space=self.action_space, obj_func=self.obj_func,
        #     #                 init_policy=self.init_target_policy, upper_bound=False)
        #     pass
        # else:
        target_actions, policy_mean, policy_log_std, log_pi, *_ = self.target_policy(
            obs=obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        )

        target_pi_qs = self.q(obs, target_actions)
        if self.share_layers:
            target_pi_qs = target_pi_qs[:, 0].unsqueeze(-1)
        mean_q = target_pi_qs
        ##upper_bound (in some way)
        target_policy_loss = (-mean_q).mean()
        self.target_policy_optimizer.zero_grad()
        target_policy_loss.backward()
        self.target_policy_optimizer.step()
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

            if not self.global_opt:
                policy_loss = (upper_bound).mean()
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
            networks = self.policy.policies + self.qfs + self.tfs
        else:
            networks = [self.policy] + self.qfs + self.tfs
        # if self.mean_update:
        networks += [self.target_policy]
        return networks

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

        # if self.mean_update:
        data["target_policy_state_dict"] = self.target_policy.state_dict()
        data["target_policy_opt_state_dict"] = self.target_policy_optimizer.state_dict()
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

        # if self.mean_update:
        self.target_policy.load_state_dict(ss["target_policy_state_dict"])
        self.target_policy_optimizer.load_state_dict(ss["target_policy_opt_state_dict"])

    def obj_func(self, states, actions, upper_bound=False):
        qs = self.q(states, actions)
        if self.share_layers:
            stds = qs[:, 1].unsqueeze(-1)
            qs = qs[:, 0].unsqueeze(-1)
        else:
            stds = self.std(states, actions)
        if upper_bound:
            obj = qs + self.standard_bound * stds
        else:
            obj = qs
        return obj

    def optimize_policies(self, buffer, out_dir='', epoch=0, save_fig=False):
        if self.ensemble:
            return
        # if self.mean_update:
        #     optimize_policy(self.target_policy, self.target_policy_optimizer, buffer,
        #                     action_space=self.action_space, obj_func=self.obj_func,
        #                     init_policy=self.init_target_policy, upper_bound=False, out_dir=out_dir,
        #                     epoch=epoch, save_fig=save_fig)
        optimize_policy(self.policy, self.policy_optimizer, buffer, obj_func=self.obj_func,
                        init_policy=self.init_policy, action_space=self.action_space, upper_bound=True,
                        out_dir=out_dir, epoch=epoch, save_fig=save_fig)
