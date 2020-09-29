import os.path as osp
from torch_geometric.datasets import Flickr, Reddit, Yelp
from dataset import PPI
from torch_geometric.data import GraphSAINTRandomWalkSampler, \
    NeighborSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, ClusterData,\
    ClusterLoader
from sampler import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, MySAINTSampler
import torch.nn as nn
from metric_and_loss import NormCrossEntropyLoss, NormBCEWithLogitsLoss
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import torch

def load_dataset(dataset='flickr'):
    """

    Args:
        dataset: str, name of dataset, assuming the raw dataset path is ./data/your_dataset/raw.
                 torch_geometric.dataset will automatically preprocess the raw files and save preprocess dataset into
                 ./data/your_dataset/preprocess

    Returns:
        dataset
    """
    path = osp.join('/mnt/data/lsdata', dataset)
    if dataset == 'flickr':
        dataset = Flickr(path)

    elif dataset == 'reddit':
        dataset = Reddit(path)

    elif dataset == 'ppi':
        dataset = PPI(path)

    elif dataset == 'ppi-large':
        dataset = PPI(path)

    elif dataset == 'yelp':
        dataset = Yelp(path)
    else:
        raise KeyError('Dataset name error')

    return dataset


def build_loss_op(args):
    if args.dataset in ['flickr', 'reddit']:
        if args.loss_norm == 1:
            return NormCrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(reduction='none')
    else:
        if args.loss_norm == 1:
            return NormBCEWithLogitsLoss()
        else:
            return nn.BCEWithLogitsLoss(reduction='none')


def build_sampler(args, data, save_dir):
    if args.sampler == 'rw-my':
        msg = 'Use GraphSaint randomwalk sampler(mysaint sampler)'
        loader = MySAINTSampler(data, batch_size=args.batch_size, sample_type='random_walk',
                                walk_length=2, sample_coverage=1000, save_dir=save_dir)
    elif args.sampler == 'node-my':
        msg = 'Use random node sampler(mysaint sampler)'
        loader = MySAINTSampler(data, sample_type='node', batch_size=args.batch_size * 3,
                                walk_length=2, sample_coverage=1000, save_dir=save_dir)
    elif args.sampler == 'rw':
        msg = 'Use GraphSaint randomwalk sampler'
        loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=4,
                                             num_steps=5, sample_coverage=1000,
                                             save_dir=save_dir)
    elif args.sampler == 'node':
        msg = 'Use GraphSaint node sampler'
        loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size,
                                       num_steps=5, sample_coverage=1000, num_workers=0, save_dir=save_dir)

    elif args.sampler == 'edge':
        msg = 'Use GraphSaint edge sampler'
        loader = GraphSAINTEdgeSampler(data, batch_size=args.batch_size,
                                       num_steps=5, sample_coverage=1000,
                                       save_dir=save_dir, num_workers=0)
    elif args.sampler == 'cluster':
        msg = 'Use cluster sampler'
        cluster_data = ClusterData(data, num_parts=args.num_parts, save_dir=save_dir)
        loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True,
                               num_workers=0)
    else:
        raise KeyError('Sampler type error')

    return loader,msg


def filter_(data, node_idx):
    """
    presumably data_n_id and new_n_id are sorted

    """
    new_data = Data()
    N = data.num_nodes
    E = data.edge_index.size(1)
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                       value=data.edge_attr, sparse_sizes=(N, N))
    new_adj, edge_idx = adj.saint_subgraph(node_idx)
    row, col, value = new_adj.coo()
    
    for key, item in data:
        if item.size(0) == N:
            new_data[key] = item[node_idx]
        elif item.size(0) == E:
            new_data[key] = item[edge_idx]
        else:
            new_data[key] = item
    
    new_data.edge_index = torch.stack([row, col], dim=0)
    new_data.num_nodes = len(node_idx)
    new_data.edge_attr = value
    return new_data

def calc_avg_loss(loss):
    if loss.dim() > 1:
        new_loss = loss.mean(dim=1, keepdim=False)
    else:
        new_loss = loss
    return new_loss
    