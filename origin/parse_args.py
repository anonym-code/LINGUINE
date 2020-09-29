import argparse
import yaml
from easydict import EasyDict


def parse_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_args(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--train_sample', type=bool, default=True)
    parser.add_argument('--eval_sample', type=bool, default=True)
    parser.add_argument('--loss_norm', type=bool, default=False)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--sampler', type=str, default='rw',
                        choices=['rw-my', 'rw', 'ns', 'node-my', 'edge', 'node', 'cluster'])
    parser.add_argument('--gcn_type', type=str, default='sage', choices=['sage', 'gat'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--save_log', type=bool, default=True)
    parser.add_argument('--save_summary', type=bool, default=True)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--meta_learning_rate', type=float, default=1e-3)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--meta_drop_out', type=float, default=0.1)
    parser.add_argument('--meta_sampler_type', type=str, default='none', choices=['normalized', 'hard', 'bernoulli',
                                                                                  'none'])
    parser.add_argument('--use_temperature', type=bool, default=False )
    args = parser.parse_args()
    if args.train_sample == False:
        args.loss_norm = False
    args = vars(args)

    config = parse_config(config_path)
    config = config[args['dataset']]
    config = EasyDict(config)

    if not config:
        raise ValueError('Please check default hparams file!')

    for k, v in args.items():
        if v is not None:
            config[k] = v

    return config


def get_log_name(args, prefix='test', use_args=None):
    if use_args is None:
        use_args = ['dataset', 'batch_size', 'train_sample', 'eval_sample', 'loss_norm', 'sampler', 'gcn_type']
    args = vars(args)
    log_name = prefix + '-' + '-'.join([arg + '=' + str(args[arg]) for arg in use_args])
    return log_name
