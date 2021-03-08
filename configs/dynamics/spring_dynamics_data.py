# %%
import os

import torch
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, islice
from oil.datasetup.datasets import IndexedDataset
from sklearn.model_selection import train_test_split
from lie_conv.datasets import SpringDynamics
import numpy as np

from forge import flags

flags.DEFINE_integer("n_train", 3000, "Number of training datapoints.")
flags.DEFINE_integer("n_test", 2000, "Number of testing datapoints.")
flags.DEFINE_integer("n_val", 2000, "Number of validation datapoints.")
flags.DEFINE_integer("n_systems", 10000, "Size of total dataset generated.")
flags.DEFINE_string(
    "data_path",
    "./datasets/ODEDynamics/SpringDynamics/",
    "Dataset is loaded from and/or downloaded to this path.",
)
flags.DEFINE_integer("sys_dim", 2, "[add description].")
flags.DEFINE_integer("space_dim", 2, "Dimension of particle system.")
flags.DEFINE_integer("data_seed", 0, "Data splits random seed.")
flags.DEFINE_integer("num_particles", 6, "Number of particles in system.")
flags.DEFINE_integer("chunk_len", 5, "Length of trajectories.")
flags.DEFINE_boolean(
    "load_preprocessed",
    True,
    "Load data already preprocessed to avoid RAM memory spike. Ensure data exists first for the chunk_lun required.",
)
flags.DEFINE_boolean(
    "nested_and_unshuffled",
    True,
    "Makes datasets nested when increasing training data size (for data efficiency curves).",
)

# ####
# from types import SimpleNamespace

# config = SimpleNamespace(**{
#     'n_train': 3000,
#     'n_test': 2000,
#     'n_val': 200,
#     'n_systems': 150000,
#     'data_path': './datasets/ODEDynamics/SpringDynamics/',
#     'sys_dim': 2,
#     'space_dim': 2,
#     'data_seed': 1,
#     'batch_size': 200,
#     'device': 3,
#     "num_particles": 6,
#     "chunk_len": 5,
# })

def split_dataset(dataset,splits, nested_and_unshuffled=False):
    """ Inputs: A torchvision.dataset DATASET and a dictionary SPLITS
        containing fractions or number of elements for each of the new datasets.
        Allows values (0,1] or (1,N] or -1 to fill with remaining.
        Example {'train':-1,'val':.1} will create a (.9, .1) split of the dataset.
                {'train':10000,'val':.2,'test':-1} will create a (10000, .2N, .8N-10000) split
                {'train':.5} will simply subsample the dataset by half."""
    # Check that split values are valid
    N = len(dataset)
    int_splits = {k:(int(np.round(v*N)) if ((v<=1) and (v>0)) else v) for k,v in splits.items()}
    assert sum(int_splits.values())<=N, "sum of split values exceed training set size, \
        make sure that they sum to <=1 or the dataset size."
    if hasattr(dataset,'stratify') and dataset.stratify!=False:
        if dataset.stratify==True:
            y = np.array([mb[-1] for mb in dataset])
        else:
            y = np.array([dataset.stratify(mb) for mb in dataset])
    else:
        y = None
    indices = np.arange(len(dataset))
    split_datasets = {}
    for split_name, split_count in sorted(int_splits.items(),reverse=True, key=lambda kv: kv[1]):
        if split_count == len(indices) or split_count==-1:
            new_split_ids = indices
            indices = indices[:0]
        else:
            strat = None if y is None else y[indices]
            indices, new_split_ids = train_test_split(indices,test_size=split_count,stratify=strat)  
        split_datasets[split_name] = IndexedDataset(dataset,new_split_ids)

    if nested_and_unshuffled: # added this basic splitting code to make train datasets nested without shuffling dataset
        assert len(int_splits) == 3 and all(x in int_splits.keys() for x in ['train', 'test', 'val']), "code handles very specific case only."
        
        indices = list(np.arange(len(dataset)))
        train_indices = indices[:int_splits['train']]
        test_indices = indices[-int_splits['test']:]
        val_indices = indices[-(int_splits['test'] + int_splits['val']):(-int_splits['test'])]

        # paranoia
        assert (len(train_indices) == int_splits['train'] and len(val_indices) == int_splits['val'] and len(test_indices) == int_splits['test'])
        assert (len(list(set(train_indices) & set(test_indices))) == 0 and len(list(set(train_indices) & set(val_indices))) == 0 and len(list(set(val_indices) & set(test_indices))) == 0)

        split_datasets = {}
        split_datasets['train'] = IndexedDataset(dataset,train_indices)
        split_datasets['val'] = IndexedDataset(dataset,val_indices)
        split_datasets['test'] = IndexedDataset(dataset,test_indices)

    return split_datasets


def load(config):

    dataset = SpringDynamics(
        n_systems=config.n_systems,
        root_dir=config.data_path,
        space_dim=config.space_dim,
        num_particles=config.num_particles,
        chunk_len=config.chunk_len,
        load_preprocessed=config.load_preprocessed,
    )

    splits = {
        "train": config.n_train,
        "val": min(config.n_train, config.n_val),
        "test": config.n_test,
    }

    with FixedNumpySeed(config.data_seed):
        datasets = split_dataset(dataset, splits, nested_and_unshuffled=config.nested_and_unshuffled)

    # if torch.cuda.is_available():
    #     device = f"cuda:{config.device}"
    # else:
    #     device = "cpu"

    # for v in datasets.values():
    #     v.tensors_to(device)

    dataloaders = {
        k: DataLoader(
            v,
            batch_size=min(config.batch_size, config.n_train),
            num_workers=0,
            shuffle=(k == "train"),
        )
        for k, v in datasets.items()
    }

    # TODO: is this used anywhere?
    dataloaders["Train"] = islice(dataloaders["train"], len(dataloaders["val"]))

    return dataloaders, f"spring_dynamics"


# %%
# data = load(config)
# print(data[0]['train'].dataset._ids[-5:])
# print(data[0]['train'].dataset[0])
# print(data[0]['train'].dataset[-1])

# print(data[0]['test'].dataset[0])
# print(data[0]['test'].dataset[-1])