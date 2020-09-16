# %%
# ---------------------------------------------------------------------------- #
#          NOTE: this script is supposedly to be used like a notebook          #
# ---------------------------------------------------------------------------- #
# TODO for SZ: refactor things into proper py tests.

import os
import forge
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import argparse

from configs.constellation.constellation import load as load_data

device = 'cuda:7'
torch.cuda.set_device(device)

config_dct = {
    "activation_function": "swish",
    "batch_norm": False,
    "batch_size": 32,
    "content_type": "pairwise_distances",
    "corner_noise": 0.1,
    "data_augmentation": False,
    "data_dir": "data/",
    "data_seed": 0,
    "dim_hidden": 16,
    "global_rotation": 0.0,
    "group": "SE2",
    "kernel_dim": 8,
    "layer_norm": False,
    "lift_samples": 1,
    "max_rotation": 0.33,
    "mean_pooling": True,
    "model_seed": 0,
    "num_heads": 8,
    "num_layers": 1,
    "pattern_drop_prob": 0.5,
    "pattern_upscale": 0.0,
    "patterns_reps": 1,
    "shuffle_corners": True,
    "test_size": 1,
    "train_size": 1,
    "location_attention": True,
    "attention_fn": "softmax",
}

config = SimpleNamespace(**config_dct)

# %%

train_loader, test_loader, _ = load_data(config)

# %%


def rotate(X, angle):
    device = X.device
    dtype = X.dtype
    rotation_matrix = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]], device=device, dtype=dtype)
    rotation = rotation_matrix.unsqueeze(0).unsqueeze(
        0).repeat(X.shape[0], X.shape[1], 1, 1)
    out = rotation.view(-1, 2, 2).bmm(X.unsqueeze(3).view(-1, 2, 1)
                                      ).view(X.shape[0], X.shape[1], 2, 1).squeeze(3)
    return out


def get_logits(model, inp):
    out = model(*inp)
    return out.logits


def get_feature(model, inp):
    out = model.encoder.featurize(inp[0])
    return out


def get_pairs(model, inp):  # how to access the group of the model directly?
    group = model.encoder.group
    Y = torch.rand((*X.shape[:2], 4))
    pairs, _, _ = group.lift(x=(inp[0][0], Y, inp[0][1].to(
        bool)), nsamples=model.encoder.liftsamples)
    return torch.abs(pairs).sum() # pairs


def test_mask_invariance(model, forw_pass_fn, data_loader, tol=1e-15):
    with torch.no_grad():
        for data in data_loader:
            X, presence, target = [d.to(device).double() for d in data]
            perturb = np.random.randint(10, 100)
            Xt = X + (1-presence.unsqueeze(-1)) * 100

            outs = []
            for X_, presence_ in [(X, presence), (Xt, presence)]:
                torch.manual_seed(0)
                out = forw_pass_fn(model, [[X_, presence_], target])
                out = out.detach().clone()
                outs.append(out)

            max_diff = torch.abs(outs[0]-outs[1]).max()
            assert (
                max_diff < tol), f"Masking invariance exceeds tolerance. Max difference is {max_diff:.3g}."

    print('Passed masking invariance test.')


def test_permutation_invariance(model, forw_pass_fn, data_loader, tol=1e-15):
    with torch.no_grad():
        for data in data_loader:
            X, presence, target = [d.to(device).double() for d in data]

            permutation = np.random.permutation(X.shape[1])
            X_perm, presence_perm = X[:, permutation], presence[:, permutation]

            outs = []
            for X_, presence_ in [(X, presence), (X_perm, presence_perm)]:
                torch.manual_seed(0)
                out = forw_pass_fn(model, [[X_, presence_], target])
                out = out.detach().clone()
                outs.append(out)

            max_diff = torch.abs(outs[0]-outs[1]).max()
            assert (
                max_diff < tol), f"Permutation invariance exceeds tolerance. Max difference is {max_diff:.3g}."

    print('Passed permutation invariance test.')

# %%
# ---------------------------------------------------------------------------- #
#                       se2_finite implementation checks                       #
# ---------------------------------------------------------------------------- #

# reset parser so forge doesn't throw errors
forge.flags._global_parser = argparse.ArgumentParser()
from configs.constellation.se2_finite_transformer import load as load_model_se2_finite


config_se2_finite_dct = config_dct.copy()
config_se2_finite_dct.update({
    "n_outputs": 4,
    "content_type": "pairwise_distances",
    "n_enc_layers": 4,
    "n_dec_layers": 4,
    "n_heads": 4,
    "layer_norm": False,
    "cn": 4,
    "similarity_fn": "softmax",
    "arch": "only_eqv_sa",  # "only_eqv_sa",
    "num_moments": 5,
})

config_se2_finite = SimpleNamespace(**config_se2_finite_dct)

model_se2_finite, _ = load_model_se2_finite(config_se2_finite)

model_se2_finite.to(device)
model_se2_finite.double()
model_se2_finite.eval()

test_permutation_invariance(model=model_se2_finite, forw_pass_fn=get_logits,
                            data_loader=train_loader, tol=1e-15)  # passes
test_mask_invariance(model=model_se2_finite, forw_pass_fn=get_logits,
                     data_loader=train_loader, tol=1e-15)  # passes


# %%
# Equivariance to rotations?

inv_error = []

# would be good if train_loader has only one data point, otherwise, we are comparing inv. errors for different inputs.
with torch.no_grad():
    for data in train_loader:
        X, presence, target = [d.to(device).double() for d in data]
        angles = np.linspace(0, 2*math.pi, 20+1)

        for angle in angles:

            outs = []
            for angle, translate in [(0, 0), (angle, 0*angle)]:

                Xt = rotate(X, angle) + translate
                out = get_logits(model_se2_finite, [[Xt, presence], target])
                out = out.detach().clone()
                outs.append(out)

            inv_error.append(torch.abs(outs[0]-outs[1]).max().item())


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(angles, inv_error,
        label=f'Rotation group size: {config_se2_finite.cn}')
ax.set_yscale('linear')
ax.legend()
ax.set_title('Invariance error in l_infinity norm')
ax.set_xlabel('Rotation')
ax.set_ylabel('Error')

# The resulting plot should show periodic behavior being close to zero at multiples of (2pi*1/cn)
# Note when its close to zero the number is 1e-17 ish.

# %%

# ---------------------------------------------------------------------------- #
#                         general implementation checks                        #
# ---------------------------------------------------------------------------- #

# reset parser so forge doesn't throw errors
forge.flags._global_parser = argparse.ArgumentParser()
from configs.constellation.eqv_transformer_model import load as load_model_general

model, _ = load_model_general(config)

model.to(device)
model.double()
model.eval()

test_mask_invariance(model=model, forw_pass_fn=get_logits,
                     data_loader=train_loader, tol=1e-15)  # passes
test_permutation_invariance(
    model=model, forw_pass_fn=get_logits, data_loader=train_loader, tol=1e-15)

# NOTE: if random offsets are on in the lifting operation, then permutation invariance won't pass despite seed being fixed,
# since the offsets differ for each point lifted. Test passes otherwise. Comment out line 378 in lieGroups.py to make it pass.

# %%
# How does invariance vary with number of lift samples?

inv_errors = []
lift_samples_p = [2] #sorted(list(set(np.logspace(0, 2.3, 50, dtype=int))))[::-1]
config_ls = SimpleNamespace(**config.__dict__)

for idx, ls in enumerate(lift_samples_p):
    torch.cuda.empty_cache()

    config_ls.lift_samples = ls
    config_ls.location_attention = True
    config_ls.dim_hidden = 16
    config_ls.kernel_dim = 2
    config_ls.num_heads = 8
    config_ls.attention_fn = 'softmax'
    config_ls.content_type = 'pairwise_distances'


    model, _ = load_model_general(config_ls)

    model.to(device)
    model.double()
    model.eval()

    inv_error = []

    # would be good if train_loader has only one data point, otherwise, we are comparing inv. errors for different inputs.
    with torch.no_grad():
        for data in train_loader:
            X, presence, target = [d.to(device).double() for d in data]
            angles = np.linspace(0, 2*math.pi, 100+1)

            for angle in angles:

                outs = []
                for angle, translate in [(0, 0), (angle, 0*angle)]:

                    torch.cuda.empty_cache()

                    Xt = rotate(X, angle) + translate
                    torch.manual_seed(0)
                    out = get_logits(model, [[Xt, torch.ones_like(presence)], target])
                    out = out.detach().clone()
                    outs.append(out)

                inv_error.append((torch.abs(outs[0]-outs[1])).max().item())

        inv_errors.append(inv_error)

    print(f'[{idx + 1}]/[{len(lift_samples_p)}]')


# %% 
# plot invariance error vs angle of rotation (incl. possible translation)

num_colors = len(lift_samples_p)

cm = plt.get_cmap('gist_rainbow')
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])

for ls, inv_error in zip(lift_samples_p[:], inv_errors[:]):
    ax.plot(angles, inv_error, label=f'Lift samples: {ls}')
ax.set_yscale('linear')
ax.legend()
ax.set_title('Invariance error in l_infinity norm')
ax.set_xlabel('Rotation')
ax.set_ylabel('Error')

# %%
# plot invariance error vs lift samples

fig, ax = plt.subplots(figsize=(10, 8))

avg_inv_errors = [np.median(e) for e in inv_errors]
ax.plot(lift_samples_p, avg_inv_errors)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Invariance error in l_infinity norm')
ax.set_xlabel('Lift samples')
ax.set_ylabel('Error')


# %%

# torch.set_printoptions(precision=10)
# # %%
# with torch.no_grad():
#     out0 = model([(X[:, :]), torch.ones_like(presence[:, :])], target)
#     # out1 = model([rotate(X[:, :3], 2*math.pi*1/2), torch.ones_like(presence[:, :3])], target)[1]

    
# # %%

# for o0, o1 in zip(out0, out1):
#     if not isinstance(o0, (list, tuple)):
#         o0, o1 = [None, o0], [None, o1]

#     print(torch.abs(o0[1] - o1[1]).max())

# # %%
# with torch.no_grad():
#     ema = model.encoder.net[1].ema

#     results = []
#     for out in [out0, out1]:
#         pairwise_g, coset_functions, mask = out[1]

#         coset_func_out = (
#                         coset_functions + ema((pairwise_g, coset_functions, mask))[1]
#                     )

#         results.append(coset_func_out)

# print(torch.abs(results[0] - results[1]).max())



# # %%

# print(torch.abs(out0[1][0] - out1[1][0]).max())



# # %%
