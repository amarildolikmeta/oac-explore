
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
from trainer.ddpg_trainer import DDPGTrainer
from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.gaussian_trainer_ts import GaussianTrainerTS
from trainer.particle_trainer_ts import ParticleTrainerTS
from trainer.particle_trainer_oac import ParticleTrainer as ParticleTrainerOAC
from networks import FlattenMlp
from rl_algorithm import BatchRLAlgorithm
import numpy as np


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


def get_policy_producer(obs_dim, action_dim, hidden_sizes, clip=True, std=None):
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
                    bias=bias,
                    std=std
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
    def q_producer(bias=None, positive=False, train_bias=True):
        return FlattenMlp(input_size=obs_dim + action_dim,
                          output_size=output_size,
                          hidden_sizes=hidden_sizes,
                          bias=bias,
                          positive=positive,
                          train_bias=train_bias)

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
        env_args['clip_state'] = variant['clip_state']
        env_args['terminal'] = variant['terminal']
        env_args['max_state'] = variant['max_state']
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
    std = None
    if variant['algorithm_kwargs']['ddpg_noisy']:
        std = variant['std']
        std = np.ones(action_dim) * std
    policy_producer = get_policy_producer(obs_dim, action_dim, hidden_sizes=[M] * N, clip=variant['clip_action'],
                                          std=std)
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
    if variant['alg'] in ['ddpg']:
        trainer = DDPGTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] in ['oac', 'sac']:
        trainer = SACTrainer(
            policy_producer,
            q_producer,
            action_space=expl_env.action_space,
            **variant['trainer_kwargs']
        )
    elif variant['alg'] == 'p-oac':
        if variant['optimistic_exp']['should_use']:
            trainer_class = ParticleTrainerOAC
            variant['trainer_kwargs']['deterministic'] = False
        else:
            trainer_class = ParticleTrainer
            variant['trainer_kwargs']['deterministic'] = not variant['stochastic']
        q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
        q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
        trainer = trainer_class(
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
    parser.add_argument('--alg', type=str, default='oac', choices=['oac', 'p-oac', 'sac', 'g-oac', 'g-tsac', 'p-tsac',
                                                                   'ddpg'])
    parser.add_argument('--no_gpu', default=False, action='store_true')
    parser.add_argument('--base_log_dir', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--layer_size', type=int, default=16)
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--mean_update', action="store_true")
    parser.add_argument('--counts', action="store_true", help="count the samples in replay buffer")
    parser.add_argument('--log_dir', type=str, default='./data')
    parser.add_argument('--suffix', type=str, default='')
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
                                                                           "harder",
                                                                           'maze',
                                                                           'maze_easy']
                        , help='only for point environment')
    parser.add_argument('--policy_lr', type=float, default=3E-4)
    parser.add_argument('--qf_lr', type=float, default=3E-4)
    parser.add_argument('--std_lr', type=float, default=3E-5)
    parser.add_argument('--sigma_noise', type=float, default=0.0)

    parser.add_argument('--std_soft_update', action="store_true")
    parser.add_argument('--clip_state', action="store_true", help='only for point environment')
    parser.add_argument('--terminal', action="store_true", help='only for point environment')
    parser.add_argument('--max_state', type=float, default=500., help='only for point environment')

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--trainer_UB', action='store_true')
    parser.add_argument('--delta', type=float, default=0.95)
    parser.add_argument('--delta_oac', type=float, default=20.53)
    parser.add_argument('--deterministic_optimistic_exp', action='store_true')
    parser.add_argument('--no_resampling', action="store_true",
                        help="Samples are removed from replay buffer after being used once")

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=2000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=1000)
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=1)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=1000)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--clip_action', dest='clip_action', action='store_true')
    parser.add_argument('--no_clip_action', dest='clip_action', action='store_false')
    parser.set_defaults(clip_action=True)
    parser.add_argument('--entropy_tuning', dest='entropy_tuning', action='store_true')
    parser.add_argument('--no_entropy_tuning', dest='entropy_tuning', action='store_false')
    parser.set_defaults(entropy_tuning=True)
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--train_bias', dest='train_bias', action='store_true')
    parser.add_argument('--no_train_bias', dest='train_bias', action='store_false')
    parser.add_argument('--should_use',  action='store_true')
    parser.add_argument('--stochastic',  action='store_true')
    parser.set_defaults(train_bias=True)
    parser.add_argument('--soft_target_tau', type=float, default=5E-3)
    parser.add_argument('--ddpg', action='store_true', help='use a ddpg version of the algorithms')
    parser.add_argument('--ddpg_noisy', action='store_true', help='use noisy exploration policy')
    parser.add_argument('--std', type=float, default=0.1, help='use noisy exploration policy for ddpg')
    parser.add_argument('--use_target_policy', action='store_true', help='use a target policy in ddpg')
    parser.add_argument('--rescale_targets_around_mean', action='store_true', help='use a target policy in ddpg')

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
                  (args.difficulty + '/' if args.domain == 'point' else '') + \
                  ('terminal' + '/' if args.terminal and  args.domain == 'point' else '') + \
                  (str(args.dim) + '/' if args.domain == 'riverswim' else '') + \
                  ('global/' if args.global_opt else '') + \
                  ('ddpg/' if args.ddpg else '') + \
                  ('mean_update_' if args.mean_update else '') + \
                  ('_priority_' if args.priority_sample else '') + \
                  ('counts/' if args.counts else '') + \
                  ('/' if args.mean_update and not args.counts else '') + \
                   args.alg + ('_std' if args.std_soft_update else '') + '_' + el + '/' +\
                   args.suffix + '/' + str(start_time) + '/'



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
        ),
        optimistic_exp={}
    )

    args = get_cmd_args()

    variant['log_dir'] = get_log_dir(args)
    if args.load_from != '':
        variant['log_dir'] = args.load_from

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
    variant['algorithm_kwargs']['trainer_UB'] = args.trainer_UB

    variant['delta'] = args.delta
    variant['std'] = args.std
    variant['optimistic_exp']['should_use'] = args.beta_UB > 0 or args.delta > 0 and not args.alg in ['p-oac', 'sac',
                                                                                                      'g-oac', 'g-tsac',
                                                                                                      'p-tsac', 'ddpg']
    if not variant['optimistic_exp']['should_use']:
        variant['optimistic_exp']['should_use'] = args.should_use
    variant['optimistic_exp']['beta_UB'] = args.beta_UB if args.alg == 'oac' else 0
    variant['optimistic_exp']['delta'] = args.delta if args.alg in ['p-oac', 'oac', 'g-oac'] else 0
    variant['optimistic_exp']['share_layers'] = False
    if args.alg in ['p-oac']:
        variant['optimistic_exp']['share_layers'] = args.share_layers
    if args.should_use and args.alg in ['p-oac']:
        variant['optimistic_exp']['delta'] = args.delta_oac
    variant['optimistic_exp']['deterministic'] = args.deterministic_optimistic_exp
    if args.alg not in ['ddpg']:
        variant['trainer_kwargs']['use_automatic_entropy_tuning'] = args.entropy_tuning
    variant['trainer_kwargs']['discount'] = args.gamma
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.qf_lr
    variant['ensemble'] = args.ensemble
    variant['n_policies'] = args.n_policies if args.ensemble else 1
    variant['n_components'] = args.n_components
    variant['priority_sample'] = False
    variant['clip_action'] = args.clip_action
    variant['stochastic'] = args.stochastic
    if args.domain == 'lqg':
        variant['clip_action'] = False
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        variant['trainer_kwargs']['std_soft_update'] = args.std_soft_update
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
            variant['trainer_kwargs']['train_bias'] = args.train_bias
            variant['trainer_kwargs']['rescale_targets_around_mean'] = args.rescale_targets_around_mean
            variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        if args.alg in ['g-oac']:
            variant['trainer_kwargs']['std_lr'] = args.std_lr



    variant['alg'] = args.alg
    variant['dim'] = args.dim
    variant['difficulty'] = args.difficulty
    variant['max_state'] = args.max_state
    variant['clip_state'] = args.clip_state
    variant['terminal'] = args.terminal
    variant['pac'] = args.pac
    variant['no_resampling'] = args.no_resampling
    variant['r_min'] = args.r_min
    variant['r_max'] = args.r_max
    variant['sigma_noise'] = args.sigma_noise


    variant['trainer_kwargs']['soft_target_tau'] = args.soft_target_tau
    variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy
    if args.alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac']:
        N_expl = variant['algorithm_kwargs']['num_expl_steps_per_train_loop']
        N_train = variant['algorithm_kwargs']['num_trains_per_train_loop']
        B = variant['algorithm_kwargs']['batch_size']
        N_updates = (N_train * B) / N_expl
        std_soft_update_prob = 2 / (N_updates * (N_updates + 1))
        variant['trainer_kwargs']['std_soft_update_prob'] = std_soft_update_prob
    if args.ddpg or args.alg == 'ddpg':
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 1
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = 4
        variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_expl_steps_per_train_loop // 4
        variant['trainer_kwargs']['use_target_policy'] = args.use_target_policy
        variant['algorithm_kwargs']['ddpg'] = args.ddpg
        if args.alg == 'ddpg':
            variant['algorithm_kwargs']['ddpg_noisy'] = True
        else:
            variant['algorithm_kwargs']['ddpg_noisy'] = args.ddpg_noisy

    #print("Prob %s" % variant['trainer_kwargs']['std_soft_update_prob'])

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
