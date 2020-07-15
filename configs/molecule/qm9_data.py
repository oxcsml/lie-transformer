import os

from torch.utils.data import DataLoader

from oil.utils.utils import FixedNumpySeed
from oil.datasetup.datasets import split_dataset

from lie_conv.datasets import QM9datasets
from corm_data.collate import collate_fn

import forge
from forge import flags

flags.DEFINE_float(
    "subsample_trainset",
    1.0,
    "Proportion or number of samples of the full trainset to use",
)
flags.DEFINE_string(
    "task",
    "homo",
    "Which task in the QM9 dataset to train on. Pass as a comma separated string",
)
flags.DEFINE_boolean(
    "recenter", False, "Recenter the positions of atoms with charge > 0"
)


def load(config, **unused_kwargs):

    with FixedNumpySeed(0):
        datasets, num_species, charge_scale = QM9datasets(
            os.path.join(config.data_dir, "qm9")
        )
        if config.subsample_trainset != 1.0:
            datasets.update(split_dataset(datasets["train"], config.subsample_trainset))

    ds_stats = datasets["train"].stats[config.task]

    if config.recenter:
        m = datasets["train"].data["charges"] > 0
        pos = datasets["train"].data["positions"][m]
        mean, std = pos.mean(dim=0), pos.std()
        for ds in datasets.values():
            ds.data["positions"] = (ds.data["positions"] - mean[None, None, :]) / std

    dataloaders = {
        key: DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            shuffle=(key == "train"),  # False,  # TODO: temp =
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
        for key, dataset in datasets.items()
    }

    return dataloaders, num_species, charge_scale, ds_stats, f"QM9_{config.task}"


if __name__ == "__main__":
    from argparse import Namespace

    print(
        load(
            config=Namespace(
                **{
                    "task": "homo",
                    "subsample_trainset": 1.0,
                    "recenter": False,
                    "batch_size": 32,
                    "data_dir": "data/",
                }
            )
        )
    )

