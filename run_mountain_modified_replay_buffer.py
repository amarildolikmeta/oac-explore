
import argparse
import torch
from utils.env_utils import domain_to_epoch
from main import run_experiment_here, get_log_dir, experiment

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
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--layer_size', type=int, default=16)
    parser.add_argument('--n_estimators', type=int, default=2)
    parser.add_argument('--share_layers', action="store_true")
    parser.add_argument('--log_dir', type=str, default='./data/')
    parser.add_argument('--max_path_length', type=int, default=320)
    parser.add_argument('--replay_buffer_size', type=float, default=1e4)
    parser.add_argument('--num_eval_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--min_num_steps_before_training', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--r_min', type=float, default=0.)
    parser.add_argument('--r_max', type=float, default=1.)

    # optimistic_exp_hyper_param
    parser.add_argument('--beta_UB', type=float, default=0.0)
    parser.add_argument('--delta', type=float, default=0.95)
    parser.add_argument('--no_resampling', action="store_true",
                        help="Samples are removed from replay buffer after being used once")

    # Training param
    parser.add_argument('--num_expl_steps_per_train_loop',
                        type=int, default=2000)
    parser.add_argument('--num_trains_per_train_loop', type=int, default=10)
    parser.add_argument('--num_train_loops_per_epoch', type=int, default=40)
    parser.add_argument('--replay_buffer_factor', type=int, default=5)

    args = parser.parse_args()

    return args

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
    variant['domain'] = 'mountain'
    variant['num_layers'] = args.num_layers
    variant['layer_size'] = args.layer_size
    variant['share_layers'] = args.share_layers
    variant['n_estimators'] = args.n_estimators if args.alg == 'p-oac' else 2
    variant['algorithm_kwargs']['num_epochs'] = domain_to_epoch(args.domain) if args.epochs <= 0 else args.epochs
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 20 * args.max_path_length
    variant['algorithm_kwargs']['batch_size'] = args.batch_size
    variant['algorithm_kwargs']['num_train_loops_per_epoch'] = args.num_train_loops_per_epoch
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = args.num_trains_per_train_loop
    variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = args.num_trains_per_train_loop * args.batch_size
    variant['algorithm_kwargs']['min_num_steps_before_training'] = (args.replay_buffer_factor -1) * args.num_trains_per_train_loop * args.batch_size
    variant['algorithm_kwargs']['batch_size'] = args.batch_size
    variant['replay_buffer_size'] = args.replay_buffer_factor * args.num_trains_per_train_loop * args.batch_size
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
    variant['no_resampling'] = args.no_resampling
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