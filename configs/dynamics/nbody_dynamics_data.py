import os

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, islice
from oil.datasetup.datasets import split_dataset

from forge import flags

##

from lie_conv.datasets import DynamicsDataset
from lie_conv.hamiltonian import HamiltonianDynamics, EuclideanK

# The code below is from lie_conv/dataset.py but with a number of changes
# to the simulation set up.


def KeplerV(positions, masses, grav_const=1):
    """Shape (bs,n,d), and (bs,n),
    Gravitational PE: -\sum_{jk} m_jm_k/{\|q_j-q_k\|}"""
    # grav_const = 1
    n = masses.shape[-1]
    row_ind, col_ind = torch.tril_indices(n, n, offset=-1)
    moments = (masses.unsqueeze(1) * masses.unsqueeze(2))[:, row_ind, col_ind]
    pair_diff = (positions.unsqueeze(1) - positions.unsqueeze(2))[:, row_ind, col_ind]
    # print(pair_diff.norm(dim=-1).min(), pair_diff.norm(dim=-1).max())
    # pair_dist = pair_diff.norm(dim=-1) + 1e-8
    pair_dist = torch.clamp(pair_diff.norm(dim=-1), min=0.15)
    potential_energy = -grav_const * (moments / pair_dist).sum(-1)
    return potential_energy


def KeplerH(z, m, grav_const=1):
    """ with shapes (bs,2nd)"""
    bs, D = z.shape  # of ODE dims, 2*num_particles*space_dim
    q = z[:, : D // 2].reshape(*m.shape, -1)
    p = z[:, D // 2 :].reshape(*m.shape, -1)
    potential_energy = KeplerV(q, m, grav_const=grav_const)
    kinetic_energy = EuclideanK(p, m)
    assert potential_energy.shape[0] == bs
    assert kinetic_energy.shape[0] == bs
    return potential_energy + kinetic_energy


class NBodyDynamics(DynamicsDataset):
    default_root_dir = os.path.expanduser("~/datasets/ODEDynamics/NBodyDynamics/")

    def __init__(
        self,
        root_dir=default_root_dir,
        train=True,
        download=True,
        n_systems=100,
        regen=False,
        chunk_len=5,
        space_dim=3,
        delta_t=0.01,
        num_particles=6,
        traj_len=200,
    ):
        super().__init__()
        self.num_particles = num_particles
        filename = os.path.join(
            root_dir,
            f"nbody_{space_dim}D_{n_systems}_particles_{num_particles}_{('train' if train else 'test')}.pz",
        )
        self.space_dim = space_dim
        self.grav_const = 0.06

        if os.path.exists(filename) and not regen:
            ts, zs, self.SysP = torch.load(filename)
        elif download:
            sim_kwargs = dict(
                traj_len=traj_len,
                delta_t=delta_t,
            )
            ts, zs, self.SysP = self.generate_trajectory_data(n_systems, sim_kwargs)
            os.makedirs(root_dir, exist_ok=True)
            print(filename)
            torch.save((ts, zs, self.SysP), filename)
        else:
            raise Exception("Download=False and data not there")
        self.sys_dim = self.SysP.shape[-1]
        self.Ts, self.Zs = self.format_training_data(ts, zs, chunk_len)

    def sample_system(self, n_systems, space_dim=3):  # removed kwarg: n_bodies=6
        """
        See DynamicsDataset.sample_system docstring
        """
        n_bodies = self.num_particles
        grav_const = self.grav_const  # hamiltonian.py assumes G = 1
        star_mass = torch.tensor([[50.0]]).expand(n_systems, -1, -1)  # 32
        star_pos = torch.tensor([[0.0] * space_dim]).expand(n_systems, -1, -1)
        star_vel = torch.tensor([[0.0] * space_dim]).expand(n_systems, -1, -1)

        planet_mass_min, planet_mass_max = 0.1, 4.5  # 2e-1 # SZ: IN used [0.02, 9]
        planet_mass_range = planet_mass_max - planet_mass_min

        planet_dist_min, planet_dist_max = (
            0.085,
            0.85,
        )  # 0.5, 4. #SZ: IN used [10, 100]. Using 0.085, 0.85 to make order of magnitude similar to spring task
        planet_dist_range = planet_dist_max - planet_dist_min

        # sample planet masses, radius vectors
        planet_masses = (
            planet_mass_range * torch.rand(n_systems, n_bodies - 1, 1) + planet_mass_min
        )
        # rho = torch.linspace(planet_dist_min, planet_dist_max, n_bodies - 1)
        # rho = rho.expand(n_systems, -1).unsqueeze(-1)
        # rho = rho + 0.3 * (torch.rand(n_systems, n_bodies - 1, 1) - 0.5) * planet_dist_range / (n_bodies - 1)
        rho = planet_dist_max * torch.rand(n_systems, n_bodies - 1, 1) + planet_dist_min
        planet_vel_magnitude = (
            1 * (grav_const * star_mass / rho).sqrt()
        )  # SZ: this is less than the escape velocity TODO: I multipled by 4 now to match IN set up, but it's larger than escape velocity??

        if space_dim == 2:
            planet_pos, planet_vel = self._init_2d(rho, planet_vel_magnitude)
        elif space_dim == 3:
            planet_pos, planet_vel = self._init_3d(rho, planet_vel_magnitude)
        else:
            raise RuntimeError("only 2-d and 3-d systems are supported")

        # import pdb; pdb.set_trace()
        perm = torch.stack([torch.randperm(n_bodies) for _ in range(n_systems)])

        pos = torch.cat([star_pos, planet_pos], dim=1)
        pos = torch.stack([pos[i, perm[i]] for i in range(n_systems)]).reshape(
            n_systems, -1
        )
        momentum = torch.cat([star_mass * star_vel, planet_masses * planet_vel], dim=1)
        momentum = torch.stack(
            [momentum[i, perm[i]] for i in range(n_systems)]
        ).reshape(n_systems, -1)
        z0 = torch.cat([pos.double(), momentum.double()], dim=-1)

        masses = torch.cat([star_mass, planet_masses], dim=1).squeeze(-1).double()
        masses = torch.stack([masses[i, perm[i]] for i in range(n_systems)])

        return z0, (masses,)

    def _init_2d(self, rho, planet_vel_magnitude):
        n_systems, n_planets, _ = rho.shape
        # sample radial vectors
        theta = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        planet_pos = torch.cat([rho * torch.cos(theta), rho * torch.sin(theta)], dim=-1)
        # get radial tangent vector, randomly flip orientation
        e_1 = torch.stack([-planet_pos[..., 1], planet_pos[..., 0]], dim=-1)
        e_1 = e_1 + torch.rand_like(planet_pos) * 0.1 * planet_pos  # offset it a bit
        flip_dir = 2 * (
            torch.bernoulli(torch.empty(n_systems, n_planets, 1).fill_(0.5)) - 0.5
        )
        e_1 = e_1 * flip_dir / e_1.norm(dim=-1, keepdim=True)

        # # try out IN networks set up...
        # e_1 = torch.stack([planet_pos[..., 1], planet_pos[..., 0]], dim=-1)
        # e_1 = e_1 / e_1.norm(dim=-1, keepdim=True)

        planet_vel = planet_vel_magnitude * e_1
        return planet_pos, planet_vel

    def _init_3d(self, rho, planet_vel_magnitude):  # TODO SZ: modify this function too.
        n_systems, n_planets, _ = rho.shape
        # sample radial vectors
        theta = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        phi = torch.acos(
            2 * torch.rand(n_systems, n_planets, 1) - 1
        )  # incorrect to sample \phi \in [0, \pi]
        planet_pos = torch.cat(
            [
                rho * torch.sin(phi) * torch.cos(theta),
                rho * torch.sin(phi) * torch.sin(theta),
                rho * torch.cos(phi),
            ],
            dim=-1,
        )

        # get radial tangent plane orthonormal basis
        e_1 = torch.stack(
            [
                torch.zeros(n_systems, n_planets),
                -planet_pos[..., 2],
                planet_pos[..., 1],
            ],
            dim=-1,
        )
        e_2 = torch.cross(planet_pos, e_1, dim=-1)
        e_1 = e_1 / e_1.norm(dim=-1, keepdim=True)
        e_2 = e_2 / e_2.norm(dim=-1, keepdim=True)

        # sample initial velocity in tangent plane
        omega = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        planet_vel = torch.cos(omega) * e_1 + torch.sin(omega) * e_2
        planet_vel = planet_vel_magnitude * planet_vel
        return planet_pos, planet_vel

    def _get_dynamics(self, sys_params):
        H = lambda t, z: KeplerH(z, *sys_params, grav_const=self.grav_const)
        return HamiltonianDynamics(H, wgrad=False)


#############


flags.DEFINE_integer("n_train", 3000, "Number of training datapoints.")
flags.DEFINE_integer("n_test", 2000, "Number of testing datapoints.")
flags.DEFINE_integer("n_val", 2000, "Number of validation datapoints.")
flags.DEFINE_integer("n_systems", 10000, "Size of total dataset generated.")
flags.DEFINE_string(
    "data_path",
    "./datasets/ODEDynamics/NBodyDynamics/",
    "Dataset is loaded from and/or downloaded to this path.",
)
flags.DEFINE_integer("sys_dim", 1, "Dimension of the feature vector y.")
flags.DEFINE_integer("space_dim", 2, "Dimension of particle system.")
flags.DEFINE_integer("data_seed", 0, "Data splits random seed.")
flags.DEFINE_integer("num_particles", 6, "Number of particles in system.")

####
# from types import SimpleNamespace

# config = SimpleNamespace(**{
#     'n_train': 3000,
#     'n_test': 2000,
#     'n_val': 200,
#     'n_systems': 105000,
#     'data_path': './datasets/ODEDynamics/NBodyDynamics/',
#     'sys_dim': 1,
#     'space_dim': 2,
#     'data_seed': 0,
#     'batch_size': 200,
#     'device': 3,
#     "num_particles": 3,
# })


def load(config):

    dataset = NBodyDynamics(
        n_systems=config.n_systems,
        root_dir=config.data_path,
        space_dim=config.space_dim,
        num_particles=config.num_particles,
    )

    splits = {
        "train": config.n_train,
        "val": min(config.n_train, config.n_val),
        "test": config.n_test,
    }

    with FixedNumpySeed(config.data_seed):
        datasets = split_dataset(dataset, splits)

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

    return dataloaders, f"nbody_dynamics"


# data = load(config)