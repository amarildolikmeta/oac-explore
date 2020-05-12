import os
import os.path as osp
import argparse
import torch
import time
import utils.pytorch_util as ptu
from replay_buffer import ReplayBuffer, ReplayBufferCount
from replay_buffer_no_resampling import ReplayBufferNoResampling
from utils.env_utils import NormalizedBoxEnv, domain_to_epoch, env_producer
from utils.rng import set_global_pkg_rng_state
from launcher_util import run_experiment_here
from path_collector import MdpPathCollector, RemoteMdpPathCollector
from trainer.policies import TanhGaussianPolicy, MakeDeterministic, EnsemblePolicy, TanhGaussianMixturePolicy
from trainer.policies import GaussianPolicy
from trainer.trainer import SACTrainer
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.gaussian_trainer_ts import GaussianTrainerTS
from trainer.particle_trainer_ts import ParticleTrainerTS
from networks import FlattenMlp
from rl_algorithm import BatchRLAlgorithm
import numpy as np
# import ray
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


def get_policy_producer(obs_dim, action_dim, hidden_sizes, clip=True):
    def policy_producer(deterministic=False, bias=None, ensemble=False, n_policies=1, n_components=1,
                        approximator=None, share_layers=False):
        if ensemble:
            policy = EnsemblePolicy(approximator=approximator,
                                    hidden_sizes=hidden_sizes,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim,
                                    n_policies=n_policies,
                                    bias=bias,
                                    share_layers=share_layers)
        else:
            if not clip:
                policy = GaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    bias=bias
                )
            elif n_components == 1:
                policy = TanhGaussianPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    bias=bias
                )

            else:
                policy = TanhGaussianMixturePolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    bias=bias,
                    n_components=n_components
                )
            '''
            policy = TanhGaussianMixturePolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                bias=bias,
                n_components=n_components
            )
            '''
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
    if domain in ['point']:
        env_args['difficulty'] = variant['difficulty']
    if 'cliff' in domain:
        env_args['sigma_noise'] = variant['sigma_noise']

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
        # n_estimators = 1
    else:
        output_size = 1
    q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
    policy_producer = get_policy_producer(obs_dim, action_dim, hidden_sizes=[M] * N, clip=variant['clip_action'])
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

    if variant['alg'] in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac'] and variant['trainer_kwargs']["counts"]:
        replay_buffer = ReplayBufferCount(
            variant['replay_buffer_size'],
            ob_space=expl_env.observation_space,
            action_space=expl_env.action_space,
            priority_sample=variant['priority_sample']
        )
    elif variant["no_resampling"] and variant['alg'] in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        replay_buffer = ReplayBufferNoResampling(
            variant['replay_buffer_size'],
            ob_space=expl_env.observation_space,
            action_space=expl_env.action_space
        )
    else:
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
    elif variant['alg'] == 'g-tsac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        q_posterior_producer = None
        if variant['share_layers']:
            q_posterior_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=1)
        trainer = GaussianTrainerTS(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            n_components=variant['n_components'],
            q_posterior_producer=q_posterior_producer,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] == 'p-tsac':
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        q_posterior_producer = None
        if variant['share_layers']:
            q_posterior_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=1)
        trainer = ParticleTrainerTS(
            policy_producer,
            q_producer,
            n_estimators=n_estimators,
            delta=variant['delta'],
            q_min=q_min,
            q_max=q_max,
            action_space=expl_env.action_space,
            n_components=variant['n_components'],
            q_posterior_producer=q_posterior_producer,
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

    if prev_exp_state is not None:
        expl_path_collector.restore_from_snapshot(
            prev_exp_state['exploration'])

        remote_eval_path_collector.restore_from_snapshot(
            prev_exp_state['evaluation_remote'])
        remote_eval_path_collector.set_global_pkg_rng_state(
            prev_exp_state['evaluation_remote_rng_state'])

        replay_buffer.restore_from_snapshot(prev_exp_state['replay_buffer'])

        trainer.restore_from_snapshot(prev_exp_state['trainer'])

        set_global_pkg_rng_state(prev_exp_state['global_pkg_rng_state'])

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
    parser.add_argument('--n_policies', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alg', type=str, default='oac', choices=['oac', 'p-oac', 'sac', 'g-oac', 'g-tsac', 'p-tsac'])
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--layer_size', type=int, default=16)
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--mean_update', action="store_true")
    parser.add_argument('--counts', action="store_true", help="count the samples in replay buffer")
    parser.add_argument('--log_dir', type=str, default='./data/')
    parser.add_argument('--max_path_length', type=int, default=100)
    parser.add_argument('--replay_buffer_size', type=float, default=1e4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--r_min', type=float, default=0.)
    parser.add_argument('--r_max', type=float, default=1.)
    parser.add_argument('--r_mellow_max', type=float, default=1.)
    parser.add_argument('--mellow_max', action="store_true")
    parser.add_argument('--priority_sample', action="store_true")
    parser.add_argument('--global_opt', action="store_true")
    parser.add_argument('--save_sampled_data', default=False, action='store_true')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--snapshot_gap', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')
    parser.add_argument('--snapshot_mode', type=str, default='last_every_gap', choices=['last_every_gap',
                                                                                        'all',
                                                                                        'last',
                                                                                        "gap",
                                                                                        'gap_and_last'])
    parser.add_argument('--difficulty', type=str, default='hard', choices=['easy',
                                                                           'medium',
                                                                           'hard',
                                                                           "harder"])
    parser.add_argument('--policy_lr', type=float, default=3E-4)
    parser.add_argument('--qf_lr', type=float, default=3E-4)
    parser.add_argument('--sigma_noise', type=float, default=0.0)
    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.95)
    parser.add_argument('--no_resampling', action="store_true",
                        help="Samples are removed from replay buffer after being used once")

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=2000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=1)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--clip_action', dest='clip_action', action='store_true')
    parser.add_argument('--no-clip_action', dest='clip_action', action='store_false')
    parser.set_defaults(clip_action=True)

    args = parser.parse_args()

    return args


def get_log_dir(args, should_include_base_log_dir=True, should_include_seed=True, should_include_domain=True):
    start_time = time.time()
    if args.load_dir != '':
        log_dir = args.load_dir
    else:
        if args.n_policies > 1:
            el = str(args.n_policies)
        elif args.n_components > 1:
            el = str(args.n_components)
        else:
            el = ''
        log_dir = args.log_dir + '/' + args.domain + '/' + \
                  ('global/' if args.global_opt else '') + \
                  ('mean_update_' if args.mean_update else '') + \
                  ('counts/' if args.counts else '') + \
                  ('/' if args.mean_update and not args.counts else '') + \
                   args.alg + '_' + el + '/' + str(start_time) + '/'

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
    variant['n_estimators'] = args.n_estimators if args.alg in ['p-oac', 'p-tsac'] else 2
    variant['replay_buffer_size'] = int(args.replay_buffer_size)

    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain) if args.epochs <= 0 else args.epochs
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_expl_steps_per_train_loop
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = args.num_eval_steps_per_epoch
    variant['algorithm_kwargs']['min_num_steps_before_training'] = args.min_num_steps_before_training
    variant['algorithm_kwargs']['batch_size'] = args.batch_size
    variant['algorithm_kwargs']['save_sampled_data'] = args.save_sampled_data
    variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_train_loops_per_epoch

    variant['delta'] = args.delta
    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0 and not args.alg in ['p-oac', 'sac',
                                                                                                      'g-oac', 'g-tsac',
                                                                                                      'p-tsac']
    variant['optimistic_exp']['beta_UB'] = args.beta_UB if args.alg == 'oac' else 0
    variant['optimistic_exp']['delta'] = args.delta if args.alg in ['p-oac', 'oac', 'g-oac'] else 0
    variant['trainer_kwargs']['discount'] = args.gamma
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.qf_lr
    variant['ensemble'] = args.ensemble
    variant['n_policies'] = args.n_policies if args.ensemble else 1
    variant['n_components'] = args.n_components
    variant['priority_sample'] = False
    variant['clip_action'] = args.clip_action
    if args.domain == 'lqg':
        variant['clip_action'] = False
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        variant['trainer_kwargs']['share_layers'] = args.share_layers
        variant['trainer_kwargs']['mean_update'] = args.mean_update
        variant['trainer_kwargs']['counts'] = args.counts
        variant['priority_sample'] = args.priority_sample
        variant['trainer_kwargs']['global_opt'] = args.global_opt
        if args.alg in ['p-oac', 'g-oac']:
            variant['trainer_kwargs']['r_mellow_max'] = args.r_mellow_max
            variant['trainer_kwargs']['mellow_max'] = args.mellow_max
            variant['algorithm_kwargs']['global_opt'] = args.global_opt
            variant['algorithm_kwargs']['save_fig'] = args.save_fig

    variant['alg'] = args.alg
    variant['dim'] = args.dim
    variant['difficulty'] = args.difficulty
    variant['pac'] = args.pac
    variant['no_resampling'] = args.no_resampling
    variant['r_min'] = args.r_min
    variant['r_max'] = args.r_max
    variant['sigma_noise'] = args.sigma_noise

    if args.no_resampling:
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 500
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 500 * args.batch_size
        variant['algorithm_kwargs']['min_num_steps_before_training'] = 4 * 500 * args.batch_size
        variant['algorithm_kwargs']['batch_size'] = args.batch_size
        variant['replay_buffer_size'] = 5 * 500 * args.batch_size
    if torch.cuda.is_available():
        gpu_id = int(args.seed % torch.cuda.device_count())
    else:
        gpu_id = None
    if not args.no_gpu:
        try:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        except:
            pass
    run_experiment_here(experiment, variant,
                        seed=args.seed,
                        use_gpu=not args.no_gpu and torch.cuda.is_available(),
                        gpu_id=gpu_id,

                        # Save the params every snapshot_gap and override previously saved result
                        snapshot_gap=args.snapshot_gap,
                        snapshot_mode=args.snapshot_mode,

                        log_dir=variant['log_dir']
                        )
