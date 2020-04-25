import json
from trainer.gaussian_trainer_ts import GaussianTrainerTS
import numpy as np
import torch
from main import env_producer, get_policy_producer, get_q_producer
from utils.core import np_ify, torch_ify
import matplotlib.pyplot as plt
from utils.pythonplusplus import load_gzip_pickle

ts = '1584884279.5007188'
iter = 190
path = 'data/riverswim/p-oac/' + ts + '/variant.json'
restore = True

variant = json.load(open(path,'r'))
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
alg = variant['alg']
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
    n_estimators = 1
else:
    output_size = 1
ob = expl_env.reset()
print(ob)
q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
policy_producer = get_policy_producer(
    obs_dim, action_dim, hidden_sizes=[M] * N)
q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
trainer = GaussianTrainerTS(
        policy_producer,
        q_producer,
        n_estimators=n_estimators,
        delta=variant['delta'],
        q_min=q_min,
        q_max=q_max,
        action_space=expl_env.action_space,
        **variant['trainer_kwargs']
    )
iter = 70
delta_iter = 10
max_iter = 71
while iter <= max_iter:
    if restore:
        experiment = './data/riverswim/' + alg + '/' + ts + '/itr_' + str(iter) + '.zip_pkl'
        exp = load_gzip_pickle(experiment)
        trainer.restore_from_snapshot(exp['trainer'])
    delta_action = 0.05
    xs = np.array([-1 + i*delta_action for i in range(int(2 / delta_action +1))])
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))
    for i in range(25):
        row = i // 5
        col = i % 5
        ob = np.array([i]).reshape((1, 1))
        action = -1
        pairs = []
        while action < 1.05:
            a = np.array([action]).reshape((1,1))
            qs, stds = trainer.predict(ob, a)
            qs, stds = np_ify(qs), np_ify(stds)
            action += delta_action
            pairs.append(np.array([qs, stds]).ravel())
        pairs = np.array(pairs)
        # mean_q = particles.mean(axis=1)
        # std_q = particles.std(axis=1)
        xs = xs[:pairs.shape[0]]
        # ax[row, col].plot(xs, mean_q)
        # ax[row, col].fill_between(xs, mean_q - 2*std_q, mean_q+2*std_q, alpha=0.3)
        ax[row, col].plot(xs, pairs[:, 0])
        ax[row, col].fill_between(xs, pairs[:, 0] - pairs[:, 1], pairs[:, 0] + pairs[:, 1], alpha=0.2)


        new_obs_actions, policy_means, policy_log_stds, log_pi, *_ = trainer.policy(
            obs=torch_ify(ob), reparameterize=True, return_log_prob=True, deterministic=trainer.deterministic,
        )
        means_ = policy_means
        print(policy_log_stds)

        #ax[row, col].axvline(x=np_ify(mean_), c='red')
        '''
        ax[row, col].axvline(x=max_action)
        # for k in range(particles.shape[-1]):
        #     xs = xs[:particles.shape[0]]
        #     ax[row, col].plot(xs, particles[:, k])
        #     if k == delta_index:
        #         max_action = xs[np.argmax(particles[:, k])]
        #         ax[row, col].axvline(x=max_action)
        ax[row,col].set_title("state-" + str(i), fontdict={'fontsize': 7})
        ax[row, col].set_ylim((q_min, q_max))
        '''

    plt.show()
    fig.savefig(domain+ "_particles_upperbound" + str(iter) + ".png")
    iter += 10