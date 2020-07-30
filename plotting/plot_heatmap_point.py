import json
import sys
import seaborn as sns
sys.path.append('../')
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.trainer import SACTrainer
import numpy as np
import torch
from main import env_producer, get_policy_producer, get_q_producer
from utils.core import np_ify, torch_ify
import matplotlib.pyplot as plt
from utils.pythonplusplus import load_gzip_pickle
import argparse
import pickle
import pandas as pd

LEGEND_FONT_SIZE = 28
AXIS_FONT_SIZE = 18
TICKS_FONT_SIZE = 14
MARKER_SIZE = 10
LINE_WIDTH = 3.0
TITLE_SIZE = 22
FIG_WIDTH = 40
FIG_HEIGHT = 30
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/mean_update_counts/riverswim/p-oac_/1584884279.5007188/')
parser.add_argument('--iter', type=int, default=70)
parser.add_argument('--max_iter', type=int, default=71)
parser.add_argument('--delta_iter', type=int, default=10)
parser.add_argument('--show', action='store_true')
parser.add_argument('--point', action='store_true')

args = parser.parse_args()

base_dir = args.path
iter = 190
restore = True
path = base_dir + 'variant.json'
variant = json.load(open(path, 'r'))
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
ensemble = variant['ensemble']
delta = variant['delta']
batch_size = variant['algorithm_kwargs']['batch_size']
alg = variant['alg']
n_estimators = variant['n_estimators']
#mean_update = variant['trainer_kwargs']['mean_update']

if seed == 0:
    np.random.seed()
    seed = np.random.randint(0, 1000000)
torch.manual_seed(seed)
np.random.seed(seed)
env_args = {}
if domain in ['riverswim']:
    env_args['dim'] = variant['dim']
expl_env = env_producer(domain, seed, **env_args)
eval_env = env_producer(domain, seed * 10 + 1, **env_args)
obs_dim = expl_env.observation_space.low.size
action_dim = expl_env.action_space.low.size

# Get producer function for policy and value functions
M = variant['layer_size']
N = variant['num_layers']
n_estimators = variant['n_estimators']

if variant['share_layers']:
    output_size = n_estimators
else:
    output_size = 1
ob = expl_env.reset()
print(ob)
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M] * N)
q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
if alg == 'p-oac':
    trainer_producer = ParticleTrainer
elif alg == 'g-oac':
    trainer_producer = GaussianTrainer
else:
    trainer_producer = SACTrainer
if alg in ['p-oac', 'g-oac']:
    variant['trainer_kwargs'].update(dict(
        n_estimators=n_estimators,
        delta=variant['delta'],
        q_min=q_min,
        q_max=q_max,
    ))
trainer = trainer_producer(
    policy_producer,
    q_producer,
    action_space=expl_env.action_space,
    **variant['trainer_kwargs']
)
iter = args.iter
delta_iter = args.delta_iter
max_iter = args.max_iter
do_heatmap = True
num_trains = variant['algorithm_kwargs']['num_trains_per_train_loop']
save_sampled_data = variant['algorithm_kwargs']['save_sampled_data']
if save_sampled_data:
    try:
        all_sampled_states = pickle.load(open(base_dir + '/sampled_states.pkl', 'rb'))

        sampled_states = np.concatenate(all_sampled_states)
    except:
        data = pd.read_csv(base_dir + '/sampled_states.csv', header=None)
        sampled_states = data.iloc[:, :2].values
   # all_sampled_actions = pickle.load(open(base_dir + '/sampled_actions.pkl', 'rb'))
else:
    print("Need sampled data! Exiting!")
    exit(1)

while iter <= max_iter:
    if do_heatmap:
        try:
            experiment = base_dir + '/itr_' + str(iter) + '.zip_pkl'
            exp = load_gzip_pickle(experiment)
            trainer.restore_from_snapshot(exp['trainer'])
            heatmap = True
        except:
            heatmap = False
    else:
        heatmap = False
    delta_action = 0.25
    delta_state = 0.25
    try:
        min_state, max_state = eval_env.bounds[0][0], eval_env.bounds[1][0]
        min_y, max_y = eval_env.bounds[0][1], eval_env.bounds[1][1]
    except :
        min_state, max_state = eval_env.observation_space.low[0], eval_env.observation_space.high[0]
        min_y, max_y = eval_env.observation_space.low[1], eval_env.observation_space.high[1]
        if save_sampled_data:
            min_y = np.min(sampled_states, axis=0)[1]
            max_y = np.max(sampled_states, axis=0)[1]
            max_v = np.max(np.abs([min_y, max_y]))
            min_y = - max_v
            max_y = max_v
            delta_action = 0.1
            delta_state = 0.1
    if heatmap:
        ys = np.array([min_y + i * delta_action for i in range(int((max_y - min_y) / delta_action + 1))])[::-1]
        xs = np.array([min_state + i * delta_state for i in range(int((max_state - min_state) / delta_action + 1))])

        data_heatmap = np.zeros((len(xs), len(ys)))
        num_samples = 10
        for i in range(len(xs)):
            for j in range(len(ys)):
                o = np.array([xs[i], ys[j], 0.])
                s = 0
                for k in range(num_samples):
                    action = eval_env.action_space.sample()
                    v = np.random.uniform(low=-5, high=5, size=2)
                    ob = np.concatenate([o, v, np.array([0.])])
                    ob = np.array([ob]).reshape((1, 6))
                    a = np.array(action).reshape((1, action.shape[-1]))
                    if alg in ['p-oac']:
                        qs, _ = trainer.predict(ob, a, all_particles=True)
                        qs = np_ify(qs)
                        qs = qs.reshape((1, -1))
                        qs = np.sort(qs, axis=1)
                        std = np.std(qs.flatten())
                    elif alg in ['g-oac']:
                        qs, upper_bound = trainer.predict(ob, a, std=True)
                        stds = qs[1]
                        stds = np_ify(stds)
                        std = stds[0]
                    else:
                        qs, stds = trainer.predict(ob, a, both_values=True)
                        stds = np_ify(stds)
                        std = stds[0]
                    s += std
                std = s / num_samples
                data_heatmap[i, j] = std
    if (iter + 1) * num_trains * batch_size > sampled_states.shape[0]:
        break
    states = sampled_states[: (iter + 1) * num_trains * batch_size]
    states = states[:, :2]

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
    # fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    #ax = fig.axes
    #data = pd.DataFrame(d)
    with sns.axes_style("white"):
        if heatmap:
            num_ticks = 9
            yticks = np.linspace(0, len(ys) - 1, num_ticks, dtype=np.int)
            # the content of labels of these yticks
            yticklabels = np.array(["%.1f" % ys[idx] for idx in yticks])
            xticks = np.linspace(0, len(xs) - 1, num_ticks, dtype=np.int)
            # the content of labels of these yticks
            xticklabels = np.array(["%.1f" % xs[idx] for idx in xticks])
            fig = plt.figure(figsize=(13, 8))
            data_heatmap = data_heatmap.T
            JG2 = (sns.heatmap(data=data_heatmap, vmin=0, yticklabels=yticklabels, xticklabels=xticklabels))
            JG2.set_yticks(yticks)
            JG2.set_xticks(xticks)
            fig.savefig(base_dir + '/heatmap_map' + str(iter) + '.png')
            plt.close(fig)
            del fig
        JG = (sns.jointplot(x=states[:, 0], y=states[:, 1], kind="hex", color="k", xlim=(min_state, max_state),
                            ylim=(min_y, max_y)).set_axis_labels("x", "y"))
        fig = JG.fig
        JG.savefig(base_dir + '/density_map' + str(iter) + '.png')
        #fig.savefig(base_dir + '/density_map' + str(iter) + '.png')
        plt.close(fig)
        states = sampled_states[(iter) * num_trains * batch_size: (iter + 1) * num_trains * batch_size]
        states = states[:, :2]
        del JG
        JG = (sns.jointplot(x=states[:, 0], y=states[:, 1], kind="hex", color="k", xlim=(min_state, max_state),
                            ylim=(min_y, max_y)).set_axis_labels("x", "y"))
        fig = JG.fig
        JG.savefig(base_dir + '/density_map_current_iter_' + str(iter) + '.png')
        plt.close(fig)

        # for A in JG.fig.axes:
        #     fig._axstack.add(fig._make_key(A), A)
        # for A in JG2.fig.axes:
        #     fig._axstack.add(fig._make_key(A), A)
        # fig.axes[0].set_position([0.05, 0.05, 0.4, 0.4])
        # fig.axes[1].set_position([0.05, 0.45, 0.4, 0.05])
        # fig.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
        # fig.axes[3].set_position([0.55, 0.05, 0.4, 0.4])
        # fig.axes[4].set_position([0.55, 0.45, 0.4, 0.05])
        # fig.axes[5].set_position([0.95, 0.05, 0.05, 0.4])
        fig = JG.fig

    # if alg == 'p-oac':
    #     delta_index = trainer.delta_index
    #     while action < eval_env.action_space.high + delta_action:
    #         a = np.array([action]).reshape((1, 1))
    #         qs, _ = trainer.predict(ob, a, all_particles=True)
    #         qs = np_ify(qs)
    #         qs = qs.reshape((1, -1))
    #         qs = np.sort(qs, axis=1)
    #         particles.append(qs)
    #         action += delta_action
    #     particles = np.concatenate(particles, axis=0)
    #     mean_q = particles.mean(axis=1)
    #     std_q = particles.std(axis=1)
    #     xs = xs[:particles.shape[0]]
    #     axis.plot(xs, mean_q)
    #     axis.fill_between(xs, mean_q - 2 * std_q, mean_q + 2 * std_q, alpha=0.3)
    #     axis.plot(xs, particles[:, delta_index])
    #     # for i in range(particles.shape[1]):
    #     #     # axis.plot(xs, mean_q)
    #     #     axis.plot(xs, particles[:, i])
    #     #     # axis.fill_between(xs, mean_q - 2*std_q, mean_q+2*std_q, alpha=0.3)
    #     #     # axis.plot(xs, particles[:, delta_index])
    #     max_action = xs[np.argmax(particles[:, delta_index])]
    #     max_action_mean = xs[np.argmax(particles.mean(axis=1))]
    #     max_action_Q = particles[np.argmax(particles[:, delta_index]), delta_index]
    #     max_action_mean_Q = mean_q[np.argmax(particles.mean(axis=1))]
    # else:
    #     qfs_array = []
    #     stds_array = []
    #     bounds_array = []
    #     while action < eval_env.action_space.high[0] + delta_action:
    #         a = np.array([action]).reshape((1, 1))
    #         qs, upper_bound = trainer.predict(ob, a, std=True)
    #         stds = qs[1]
    #         qs = qs[0]
    #         qs = np_ify(qs)
    #         stds = np_ify(stds)
    #         upper_bound = np_ify(upper_bound)
    #         qfs_array.append(qs)
    #         stds_array.append(stds)
    #         bounds_array.append(upper_bound)
    #         action += delta_action
    #     qfs_array = np.concatenate(qfs_array, axis=0)
    #     stds_array = np.concatenate(stds_array, axis=0)
    #     bounds_array = np.concatenate(bounds_array, axis=0).flatten()
    #     mean_q = qfs_array.flatten()
    #     std_q = stds_array.flatten()
    #     xs = xs[:qfs_array.shape[0]]
    #     axis.plot(xs, mean_q)
    #     axis.fill_between(xs, mean_q - 2 * std_q, mean_q + 2 * std_q, alpha=0.3)
    #     axis.plot(xs, bounds_array)
    #     max_action = xs[np.argmax(bounds_array)]
    #     max_action_mean = xs[np.argmax(mean_q)]
    #     max_action_Q = bounds_array[np.argmax(bounds_array)]
    #     max_action_mean_Q = mean_q[np.argmax(mean_q)]
    # if ensemble:
    #     for p in range(len(trainer.policy.policies)):
    #         new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.policy.policies[p](
    #             obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic
    #         )
    #         axis.axvline(x=np_ify(new_obs_actions), c='red', label='opt_policy')
    # else:
    #     new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.policy(
    #         obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic
    #     )
    #     axis.axvline(x=np_ify(new_obs_actions), c='red', label='opt_policy' if i == 0 else None)
    # if mean_update:
    #     new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = trainer.target_policy(
    #         obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic)
    #     axis.axvline(x=np_ify(new_obs_actions), c='green', label='Mean policy' if i == 0 else None)
    # # axis.axvline(x=max_action, c='blue', label='max_upper_bound' if i == 0 else None)
    # # axis.axvline(x=max_action_mean, c='orange', label='max_mean' if i == 0 else None)
    # axis.scatter(x=max_action, y=max_action_Q, c='blue', marker='x')
    # axis.scatter(x=max_action_mean, y=max_action_mean_Q, c='orange', marker='x')
    # # for k in range(particles.shape[-1]):
    # #     xs = xs[:particles.shape[0]]
    # #     axis.plot(xs, particles[:, k])
    # #     if k == delta_index:
    # #         max_action = xs[np.argmax(particles[:, k])]
    # #         axis.axvline(x=max_action)
    # axis.set_title("state-" + str(i), fontdict={'fontsize': TITLE_SIZE})
    # axis.tick_params(labelsize=TICKS_FONT_SIZE)
    # # axis.set_ylim((q_min, q_max))
    # if save_sampled_data:
    #     states = np.concatenate(sampled_states)
    #     actions = np.concatenate(sampled_actions)
    #     ax[-2].hist(states)
    #     ax[-1].hist(actions)
    #     ax[-2].set_title("sampled states", fontdict={'fontsize': TITLE_SIZE})
    #     ax[-1].set_title("sampled actions", fontdict={'fontsize': TITLE_SIZE})
    #     ax[-2].set_xlabel("sampled states", fontdict={'fontsize': AXIS_FONT_SIZE})
    #     ax[-1].set_ylabel("sampled actions", fontdict={'fontsize': AXIS_FONT_SIZE})
    #     ax[-2].tick_params(labelsize=TICKS_FONT_SIZE)
    #     ax[-1].tick_params(labelsize=TICKS_FONT_SIZE)

    # fig.legend()
    # if args.show:
    #     plt.show()
    iter += delta_iter
