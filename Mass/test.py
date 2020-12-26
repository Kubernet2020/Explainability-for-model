import pandas as pd
import numpy as np
import tensorflow as tf
from nbdt.graph import build_minimal_wordnet_graph, build_random_graph, \
    prune_single_successor_nodes, write_graph, get_wnids, generate_fname, \
    get_parser, get_wnids_from_dataset, get_directory, get_graph_path_from_args, \
    augment_graph, get_depth, build_induced_graph, read_graph, get_leaves, \
    get_roots, synset_to_wnid, wnid_to_name, get_root
from nbdt.hierarchy import print_graph_stats, assert_all_wnids_in_graph
from nbdt.utils import DATASETS, METHODS, Colors, fwd
from nbdt.hierarchy import generate_hierarchy_vis
from nbdt.models.myMPLnet import myMLP
from nbdt.models.Bofan_model import MLP, prepare_data, train_model
import torch
import os

#Mass
filepath = os.path.join('nbdt/models/', 'myMLP_model.pth.tar')
model = torch.load(filepath)
#Bofan
# train_dl, test_dl = prepare_data('IDS2017.csv')
# model = MLP(19)
# train_model(train_dl, model)

wnids = get_wnids_from_dataset('IDS2017')
dataset = 'IDS2017'
checkpoint=None
arch = model
induced_linkage='ward'
induced_affinity='euclidean'
branching_factor=2
no_prune=False
extra=0
method = 'induced'
seed=0
fname=''
path=''
single_path=False

# print(model.state_dict())

G = build_induced_graph(wnids,
            dataset=dataset,
            checkpoint=checkpoint,
            model=arch,
            linkage=induced_linkage,
            affinity=induced_affinity,
            branching_factor=branching_factor,
            state_dict=model.state_dict() if model is not None else None)

print_graph_stats(G, 'matched')
assert_all_wnids_in_graph(G, wnids)

if not no_prune:
    G = prune_single_successor_nodes(G)
    print_graph_stats(G, 'pruned')
    assert_all_wnids_in_graph(G, wnids)

if extra > 0:
    G, n_extra, n_imaginary = augment_graph(G, extra, True)
    print(f'[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}')
    print_graph_stats(G, 'extra')
    assert_all_wnids_in_graph(G, wnids)

path = get_graph_path_from_args(
    dataset=dataset,
    method=method,
    seed=seed,
    branching_factor=branching_factor,
    extra=extra,
    no_prune=no_prune,
    fname=fname,
    path=path,
    single_path=single_path,
    induced_linkage=induced_linkage,
    induced_affinity=induced_affinity,
    checkpoint=checkpoint,
    arch='myMLP'
)
write_graph(G, path)

from nbdt.graph import get_parser

parser = get_parser()
args = parser.parse_args()

generate_hierarchy_vis(args)

Colors.green('==> Wrote tree to {}'.format(path))