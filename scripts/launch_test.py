import json
import sys
sys.path.append("../")
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.trainer import SACTrainer
import numpy as np
import torch
from main import env_producer, get_policy_producer, get_q_producer
from utils.core import np_ify, torch_ify
import matplotlib.pyplot as plt
from utils.pythonplusplus import load_gzip_pickle
ts = '1584884279.5007188'
ts = '1586771528.641106'
iter = 190
path = '../data/point/g-oac_/' + ts
restore = True

variant = json.load(open(path + '/variant.json', 'r'))
domain = variant['domain']
seed = variant['seed']
r_max = variant['r_max']
ensemble = variant['ensemble']
delta = variant['delta']
n_estimators = variant['n_estimators']
quantiles = [i * 1. / (n_estimators - 1) for i in range(n_estimators)]
for p in range(n_estimators):
    if quantiles[p] == delta:
        delta_index = p
        break
    if quantiles[p] > delta:
        delta_index = p - 1
        break
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
alg_to_trainer = {
    'sac': SACTrainer,
    'oac': SACTrainer,
    'p-oac': ParticleTrainer,
    'g-oac': GaussianTrainer
}
trainer = alg_to_trainer[variant['alg']]

trainer = trainer(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            ensemble=variant['ensemble'],
            n_policies=variant['n_policies'],
            **variant['trainer_kwargs']
        )
experiment = path + '/params.zip_pkl'
exp = load_gzip_pickle(experiment)
trainer.restore_from_snapshot(exp['trainer'])

for i in range(10):
    s = expl_env.reset()
    done = False
    ret = 0
    t = 0
    while not done and t < 50:
        expl_env.render()
        a, agent_info = trainer.policy.get_action(s, deterministic=True)
        s, r, done, _ = expl_env.step(a)
        t += 1
        ret += r
    expl_env.render()
    print("Return: ", ret)
    input()