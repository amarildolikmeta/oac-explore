import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from trainer.trainer import SACTrainer
import utils.pytorch_util as ptu
from utils.eval_util import create_stats_ordered_dict
from typing import Iterable
from utils.core import torch_ify, np_to_pytorch_batch, optimize_policy
from utils.misc import mellow_max, reorder_and_match

class ParticleTrainer(SACTrainer):
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
            train_bias=True,
            lb=0.1 ):
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
                self.delta_index = p
                break
        for p in range(n_estimators):
            if quantiles[p] == lb:
                self.lb_index = p
                break
            if quantiles[p] > lb:
                self.lb_index = p
                break
        initial_values = np.linspace(q_min, q_max, n_estimators)
        self.share_layers = share_layers
        self.num_particles = n_estimators
        if share_layers:
            n_estimators = 1
        self.q_min = q_min
        self.q_max = q_max
        self.delta = delta
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

        self.action_space = action_space
        if share_layers:
            for i in range(n_estimators):
                self.qfs.append(q_producer(bias=initial_values, train_bias=train_bias))
                self.tfs.append(q_producer(bias=initial_values, train_bias=train_bias))
                self.qf_optimizers.append(optimizer_class(
                    self.qfs[i].parameters(),
                    lr=qf_lr, ))
        else:
            for i in range(n_estimators):
                self.qfs.append(q_producer(bias=initial_values[i], train_bias=train_bias))
                self.tfs.append(q_producer(bias=initial_values[i], train_bias=train_bias))
                self.qf_optimizers.append(optimizer_class(
                    self.qfs[i].parameters(),
                    lr=qf_lr,))
        # self.ensemble = ensemble
        # self.n_policies = n_policies
        # if ensemble:
        #     if action_space.shape[0] > 1:
        #         initial_actions = np.random.uniform(low=action_space.low,
        #                                             high=action_space.high,
        #                                             size=(n_policies, action_space.shape[0])).flatten()
        #     else:
        #         initial_actions = np.linspace(-1., 1., n_policies)
        #         initial_actions[0] += 5e-2
        #         initial_actions[-1] -= 5e-2
        #     self.policy = policy_producer(bias=initial_actions, ensemble=ensemble, n_policies=n_policies,
        #                                   approximator=self, share_layers=share_layers)
        #     self.policy_optimizers = []
        #     if share_layers:
        #         self.policy_optimizers.append(optimizer_class(
        #             self.policy.policies[0].parameters(),
        #             lr=policy_lr))
        #     else:
        #         for i in range(self.n_policies):
        #             self.policy_optimizers.append(optimizer_class(
        #                 self.policy.policies[i].parameters(),
        #                 lr=policy_lr))
        # elif self.global_opt:
        #     self.init_policy = policy_producer()
        # # if mean_update:
        # self.target_policy = policy_producer()
        # if self.global_opt:
        #     self.init_target_policy = policy_producer()
        # self.target_policy_optimizer = optimizer_class(
        #         self.target_policy.parameters(),
        #         lr=policy_lr)

    def predict(self, obs, action, all_particles=False, upper_bound=True, beta_UB=None):
        if not isinstance(obs, torch.Tensor):
            obs = np.array(obs)
            obs = torch_ify(obs)
            action = torch_ify(action)

        qs = [q(obs, action) for q in self.qfs]

        delta_index = self.delta_index
        qs = torch.stack(qs, dim=0)
        if self.share_layers:
            qs = qs.permute(2, 1, 0)
        if upper_bound:
            sorted_qs = torch.sort(qs, dim=0)[0]
            out = sorted_qs[delta_index]
        else:
            out = torch.mean(qs, dim=0)
        if all_particles:
            sorted_qs = torch.sort(qs, dim=0)[0]
            return sorted_qs, out
        return out

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
        q_preds = []
        for i in range(len(self.qfs)):
            q_preds.append(self.qfs[i](obs, actions))

        qs = torch.stack(q_preds, dim=0)
        if self.share_layers:
            qs = qs.permute(2, 1, 0)
        sorted_qs, qs_indexes = torch.sort(qs, dim=0)
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
                obs=next_obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
            )
        normal_order = torch.stack([torch.ones(qs.shape[1]) * i for i in range(qs.shape[0])], dim=0)
        num_current_out_of_order = torch.sum(torch.squeeze(qs_indexes) != normal_order)
        target_qs = [q(next_obs, new_next_actions) for q in self.tfs]
        target_qs = torch.stack(target_qs, dim=0)
        if self.share_layers:
            target_qs = target_qs.permute(2, 1, 0)
        target_qs_sorted, target_qs_indexes = torch.sort(target_qs, dim=0)
        num_target_out_of_order = torch.sum(torch.squeeze(target_qs_indexes) != normal_order)
        # target_q_values = torch.min(target_qs, dim=0)[0] - alpha * new_log_pi
        target_q_values = target_qs_sorted
        # target_q_values = target_qs
        q_target = self.reward_scale * rewards + \
                   (1. - terminals) * discount * target_q_values

        if self.std_soft_update:
            current_q_values = sorted_qs.detach()
            current_q_mean = torch.mean(current_q_values, dim=0)

            next_q_values = self.reward_scale * rewards + (1. - terminals) * discount * target_q_values
            next_q_values = next_q_values.detach()
            next_q_mean = torch.mean(next_q_values, dim=0)

            q_target = self.std_soft_update_prob * next_q_values + \
                       (1 - self.std_soft_update_prob) * (current_q_values - current_q_mean + next_q_mean)
        if self.counts:
            counts = batch['counts']
            factor = torch.zeros_like(counts)
            factor[counts == 0] = 1
            q_target = (q_target * factor) + (1 - factor) * (sorted_qs - torch.mean(sorted_qs, dim=0) + torch.mean(q_target, dim=0))
            #assert ((q_target_2.std(dim=0) - q_target.std(dim=0)) == 0).all()

        qf_losses = []
        qf_loss = 0

        ## Do the inverse ordering to give each head the correct targets wrt
        # the specific quantile they represent for each sample in the batch

        #targets_1 = reorder_and_match(q_target, qs_indexes)
        #targets = targets_1

        #assert (torch.gather(targets_1, 0, qs_indexes) - targets).isclose(torch.Tensor(0)).all()
        qs = sorted_qs    #TODO questo e solo per prova rimuovi commento per riordinare
        targets = q_target

        #rescale targets
        # q_range = targets[-1] - targets[0]
        # factor = torch.ones_like(q_range)
        # factor[q_range > self.q_max - self.q_min] = 0
        # max_spread = self.q_max - self.q_min
        # q_mean = torch.mean(targets, dim=0)
        # targets = factor * targets + (1 - factor) * ((targets - q_mean) * (max_spread / (q_range + 1e-6)) + q_mean)
        if self.share_layers:
            for i in range(self.num_particles):
                q_loss = self.qf_criterion(qs[i], targets[i].detach())
                qf_losses.append(q_loss)
                qf_loss += q_loss
            self.qf_optimizers[0].zero_grad()
            if torch.isnan(qf_loss).any():
                print("What")
            qf_loss.backward(retain_graph=True)
            self.qf_optimizers[0].step()
        else:
            for i in range(self.num_particles):
                q_loss = self.qf_criterion(qs[i], targets[i].detach())
                qf_losses.append(q_loss)
                qf_loss += q_loss
                self.qf_optimizers[i].zero_grad()
                q_loss.backward(retain_graph=True)
                self.qf_optimizers[i].step()

        """
        Policy and Alpha Loss
        """


        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs=obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
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

        pi_qs = [q(obs, new_obs_actions) for q in self.qfs]
        delta_index = self.delta_index
        pi_qs = torch.stack(pi_qs, dim=0)
        if self.share_layers:
            pi_qs = pi_qs.permute(2, 1, 0)
        sorted_qs = torch.sort(pi_qs, dim=0)[0]
        q_new_actions = sorted_qs[0]
        # q_new_actions = torch.mean(sorted_qs, dim=0)
        # q_new_actions = torch.min(qs, dim=0)[0]
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        if torch.isnan(policy_loss).any():
            print("What")
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # target_actions, policy_mean, policy_log_std, log_pi, *_ = self.target_policy(
        #     obs=obs, reparameterize=True, return_log_prob=True, deterministic=self.deterministic
        # )
        #
        # target_pi_qs = [q(obs, target_actions) for q in self.qfs]
        # target_pi_qs = torch.stack(target_pi_qs, dim=0)
        # if self.share_layers:
        #     target_pi_qs = target_pi_qs.permute(2, 1, 0)
        # mean_q = torch.mean(target_pi_qs, dim=0)
        # ##upper_bound (in some way)
        # target_policy_loss = (-mean_q).mean()
        # self.target_policy_optimizer.zero_grad()
        # target_policy_loss.backward()
        # self.target_policy_optimizer.step()

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
            self.eval_statistics['QF mean'] = np.mean(ptu.get_numpy(qs), axis=0).mean()
            self.eval_statistics['QF std'] = np.std(ptu.get_numpy(qs), axis=0).mean()
            self.eval_statistics['QF Unordered'] = ptu.get_numpy(num_current_out_of_order).mean()
            self.eval_statistics['QF target Undordered'] = ptu.get_numpy(num_target_out_of_order).mean()

            for i in range(self.num_particles):
                self.eval_statistics['QF' + str(i) + ' Loss'] = np.mean(ptu.get_numpy(qf_losses[i]))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q' + str(i) + 'Predictions',
                    ptu.get_numpy(qs[i]),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q' + str(i) + 'Targets',
                    ptu.get_numpy(target_qs[i]),
                ))
            if not self.global_opt:
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

        networks = [self.policy] + self.qfs + self.tfs
        # # if self.mean_update:
        # networks += [self.target_policy]
        return networks

    def get_snapshot(self):
        data = dict(
            policy_state_dict=self.policy.state_dict(),
            policy_optim_state_dict=self.policy_optimizer.state_dict(),

            eval_statistics=self.eval_statistics,
            _n_train_steps_total=self._n_train_steps_total,
            _need_to_update_eval_statistics=self._need_to_update_eval_statistics
            )
        if self.use_automatic_entropy_tuning:
            data['alpha_optim_state_dict'] = self.alpha_optimizer.state_dict()
            data['log_alpha'] = self.log_alpha
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
        # # if self.mean_update:
        # data["target_policy_state_dict"] = self.target_policy.state_dict()
        # data["target_policy_opt_state_dict"] = self.target_policy_optimizer.state_dict()
        return data

    def restore_from_snapshot(self, ss):
        policy_state_dict, policy_optim_state_dict = ss['policy_state_dict'], ss['policy_optim_state_dict']

        self.policy.load_state_dict(policy_state_dict)
        self.policy_optimizer.load_state_dict(policy_optim_state_dict)

        qfs_state_dicts, qfs_optims_state_dicts = ss['qfs_state_dicts'], ss['qfs_optims_state_dicts']
        target_qfs_state_dicts = ss['target_qfs_state_dicts']
        for i in range(len(qfs_state_dicts)):
            self.qfs[i].load_state_dict(qfs_state_dicts[i])
            self.qf_optimizers[i].load_state_dict(qfs_optims_state_dicts[i])
            self.tfs[i].load_state_dict(target_qfs_state_dicts[i])
        if self.use_automatic_entropy_tuning:
            log_alpha, alpha_optim_state_dict = ss['log_alpha'], ss['alpha_optim_state_dict']
            self.log_alpha.data.copy_(log_alpha)
            self.alpha_optimizer.load_state_dict(alpha_optim_state_dict)
        self.eval_statistics = ss['eval_statistics']
        self._n_train_steps_total = ss['_n_train_steps_total']
        self._need_to_update_eval_statistic = ss['_need_to_update_eval_statistics']
        # # if self.mean_update:
        # self.target_policy.load_state_dict(ss["target_policy_state_dict"])
        # self.target_policy_optimizer.load_state_dict(ss["target_policy_opt_state_dict"])

    def obj_func(self, states, actions, upper_bound=False):
        pi_qs = [q(states, actions) for q in self.qfs]
        pi_qs = torch.stack(pi_qs, dim=0)
        if self.share_layers:
            pi_qs = pi_qs.permute(2, 1, 0)
        if upper_bound:
            sorted_qs = torch.sort(pi_qs, dim=0)[0]
            # q_new_actions = torch.min(qs, dim=0)[0]
            obj = sorted_qs[self.delta_index]
        else:
            obj = torch.mean(pi_qs, dim=0)
        return obj

    def optimize_policies(self, buffer, out_dir='', epoch=0,save_fig=False):
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


