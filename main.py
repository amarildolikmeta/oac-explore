import os
import os.path as osp
import argparse
import torch
import time
import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic, EnsemblePolicy
from trainer.trainer import SACTrainer
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from networks import FlattenMlp
from rl_algorithm import BatchRLAlgorithm
import numpy as np
import ray
import logging
# ray.init(
#     # If true, then output from all of the worker processes on all nodes will be directed to the driver.
#     log_to_driver=True,
#     logging_level=logging.WARNING,
#
#     # # The amount of memory (in bytes)
#     # object_store_memory=1073741824, # 1g
#     # redis_max_memory=1073741824 # 1g
# )


def get_current_branch(dir):

    from git import Repo

    repo = Repo(dir)
    return repo.active_branch.name


def get_policy_producer(obs_dim, action_dim, hidden_sizes):

    def policy_producer(deterministic=False, bias=None, ensemble=False, n_policies=1,
                    approximator=None):
        if ensemble:
            policy = EnsemblePolicy(approximator=approximator,
                                    hidden_sizes=hidden_sizes,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    n_policies=n_policies,
                                    bias=bias)
        else:
            policy = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                bias=bias
            )
            if deterministic:
                policy = MakeDeterministic(policy)

        return policy

    return policy_producer


def get_q_producer(obs_dim, action_dim, hidden_sizes, output_size=1):
    def q_producer(bias=None, positive=False):
        return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=output_size,
                          hidden_sizes=hidden_sizes,
                          bias=bias,
                          positive=positive)

    return q_producer


def experiment(variant, prev_exp_state=None):

    domain = variant['domain']
    seed = variant['seed']
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
    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
    policy_producer = get_policy_producer(
        obs_dim, action_dim, hidden_sizes=[M] * N)
    # Finished getting producer

    remote_eval_path_collector = MdpPathCollector(
        eval_env
    )

    # remote_eval_path_collector = RemoteMdpPathCollector.remote(
    #     domain, seed * 10 + 1,
    #     policy_producer
    # )

    expl_path_collector = MdpPathCollector(
        expl_env,
    )
    replay_buffer = ReplayBuffer(
        variant['replay_buffer_size'],
        ob_space=expl_env.observation_space,
        action_space=expl_env.action_space
    )
    if variant['alg'] in ['oac', 'sac']:
        trainer = SACTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] == 'p-oac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        trainer = ParticleTrainer(
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
    elif variant['alg'] == 'g-oac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        trainer = GaussianTrainer(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            pac=variant['pac'],
            ensemble=variant['ensemble'],
            n_policies=variant['n_policies'],
            **variant['trainer_kwargs']
        )
    else:
        raise ValueError("Algorithm no implemented:" + variant['alg'])

    algorithm = BatchRLAlgorithm(
        trainer=trainer,
        exploration_data_collector=expl_path_collector,
        remote_eval_data_collector=remote_eval_path_collector,
        replay_buffer=replay_buffer,
        optimistic_exp_hp=variant['optimistic_exp'],
        deterministic=variant['alg'] == 'p-oac',
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    # if prev_exp_state is not None:
    #
    #     expl_path_collector.restore_from_snapshot(
    #         prev_exp_state['exploration'])
    #
    #     ray.wait([remote_eval_path_collector.restore_from_snapshot.remote(
    #         prev_exp_state['evaluation_remote'])])
    #     ray.wait([remote_eval_path_collector.set_global_pkg_rng_state.remote(
    #         prev_exp_state['evaluation_remote_rng_state']
    #     )])
    #
    #     replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])
    #
    #     trainer.restore_from_snapshot(prev_exp_state['trainer'])
    #
    #     set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

    start_epoch = prev_exp_state['epoch'] + \
        1 if prev_exp_state is not None else 0

    algorithm.train(start_epoch)


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--domain', type=str, default='mountain')
    parser.add_argument('--dim', type=int, default=25)
    parser.add_argument('--pac', action="store_true")
    parser.add_argument('--ensemble', action="store_true")
    parser.add_argument('--n_policies', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alg', type=str, default='oac', choices=['oac', 'p-oac', 'sac', 'g-oac',])
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--layer_size', type=int, default=16)
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--log_dir', type=str, default='./data/')
    parser.add_argument('--max_path_length', type=int, default=200)
    parser.add_argument('--replay_buffer_size', type=float, default=1e4)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--r_min', type=float, default=0.)
    parser.add_argument('--r_max', type=float, default=1.)

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.95)

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=2000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)

    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):

    log_dir = '../data/master/num_expl_steps_per_train_loop_1000_num_trains_per_train_loop_1000/ beta_UB_4.66_delta_23.53/mountain/seed_0'
    start_time = time.time()
    log_dir = args.log_dir + args.domain + '/' + args.alg + '/' + str(start_time) + '/'
    # # #       ''osp.join(
    # # # get_current_branch('./'),
    # #
    # #     # Algo kwargs portion
    # #     'num_expl_steps_per_train_loop_' + str(args.num_expl_steps_per_train_loop) + '_num_trains_per_train_loop_' +
    # #     str(args.num_trains_per_train_loop),
    # #
    # #     # optimistic exploration dependent portion
    # #    ' beta_UB_' + str(args.beta_UB) + '_delta_' + str(args.delta),
    # # )
    # #
    # # if should_include_domain:
    # #     log_dir = osp.join(log_dir, args.domain)
    # #
    # # if should_include_seed:
    # #     log_dir = osp.join(log_dir, 'seed_' + str(args.seed))
    # #
    # # if should_include_base_log_dir:
    #     log_dir = osp.join(args.base_log_dir, log_dir)

    return log_dir


if __name__ == "__main__":

    # Parameters for the experiment are either listed in variant below
    # or can be set through cmdline args and will be added or overrided
    # the corresponding attributein variant

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E4),
        algorithm_kwargs=dict(
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=None,
            num_expl_steps_per_train_loop=None,
            min_num_steps_before_training=1000,
            max_path_length=100,
            batch_size=32,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        optimistic_exp={}
    )

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)

    variant['seed'] = args.seed
    variant['domain'] = args.domain
    variant['num_layers'] = args.num_layers
    variant['layer_size'] = args.layer_size
    variant['share_layers'] = args.share_layers
    variant['n_estimators'] = args.n_estimators if args.alg == 'p-oac' else 2
    variant['replay_buffer_size'] = int(args.replay_buffer_size)

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain)
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = args.num_eval_steps_per_epoch
    variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_num_steps_before_training
    variant['algorithm_kwargs']['batch_size'] = args.batch_size

    variant['delta'] = args.delta
    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0 and not args.alg in ['p-oac', 'sac',
                                                                                                      'g-oac']
    variant['optimistic_exp']['beta_UB'] = args.beta_UB if args.alg == 'oac' else 0
    variant['optimistic_exp']['delta'] = args.delta if args.alg in ['p-oac', 'oac', 'g-oac'] else 0

    variant['trainer_kwargs']['discount'] = args.gamma
    variant['ensemble'] = args.ensemble
    variant['n_policies'] = args.n_policies if args.ensemble else 1

    variant['alg'] = args.alg
    variant['dim'] = args.dim
    variant['pac'] = args.pac
    variant['r_min'] = args.r_min
    variant['r_max'] = args.r_max

    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None

    run_experiment_here(experiment, variant,
                        seed=args.seed,
                        use_gpu=not args.no_gpu and torch.cuda.is_available(),
                        gpu_id=gpu_id,

                        # Save the params every snapshot_gap and override previously saved result
                        snapshot_gap=100,
                        snapshot_mode='last_every_gap',

                        log_dir=variant['log_dir']

                        )
