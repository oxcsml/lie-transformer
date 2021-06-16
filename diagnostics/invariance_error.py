# %%
import forge
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from types import SimpleNamespace
from configs.constellation.constellation import load as load_data

# reset parser so forge doesn't throw errors
forge.flags._global_parser = argparse.ArgumentParser()
from configs.constellation.eqv_transformer_model import load as load_model

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    }
)

device = "cuda:0"
torch.cuda.set_device(device)

config_dct = {
    "activation_function": "swish",
    "batch_norm": False,
    "batch_size": 1,
    "content_type": "constant",
    "corner_noise": 0.1,
    "data_augmentation": False,
    "data_dir": "data/",
    "data_seed": 0,
    "dim_hidden": 8,
    "global_rotation": 0.0,
    "group": "SE2",
    "kernel_dim": 2,
    "layer_norm": False,
    "lift_samples": 1,
    "max_rotation": 0.33,
    "mean_pooling": True,
    "model_seed": 0,
    "num_heads": 2,
    "num_layers": 1,
    "pattern_drop_prob": 0.5,
    "pattern_upscale": 0.0,
    "patterns_reps": 1,
    "shuffle_corners": True,
    "test_size": 1,
    "train_size": 1,
    "naug": 1,
    "location_attention": True,
    "attention_fn": "dot_product",
    "block_norm": "none",
    "kernel_norm": "none",
    "output_norm": "none",
    "kernel_type": "mlp",
    "architecture": "model_1",
    "debug_group_config": None,
}

config = SimpleNamespace(**config_dct)


def rotate(X, angle):
    device = X.device
    dtype = X.dtype
    rotation_matrix = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        device=device,
        dtype=dtype,
    )
    rotation = (
        rotation_matrix.unsqueeze(0).unsqueeze(0).repeat(X.shape[0], X.shape[1], 1, 1)
    )
    out = (
        rotation.view(-1, 2, 2)
        .bmm(X.unsqueeze(3).view(-1, 2, 1))
        .view(X.shape[0], X.shape[1], 2, 1)
        .squeeze(3)
    )
    return out


def get_logits(model, inp):
    out = model(*inp)
    return out.logits


def analyze_invariance(lift_samples_p, num_runs=5, fix_seed_before_forw_pass=False, tol=7e-8):
    group = "SE2"
    debug_group_config = {
        "ensure_thetas_in_range": True,
        "add_random_offsets": True,
        "tol": tol,
    }
    rng = np.random.RandomState(seed=0)

    inv_errors_multiple_runs = []
    for idx_run in range(num_runs):
        inv_errors = []

        config_ls = SimpleNamespace(**config.__dict__)
        for idx, ls in enumerate(lift_samples_p):
            torch.cuda.empty_cache()

            config_ls.lift_samples = ls
            config_ls.group = group
            config_ls.debug_group_config = debug_group_config
            config_ls.model_seed = rng.randint(0, 10000)
            config_ls.data_seed = rng.randint(0, 10000)

            train_loader, _, _ = load_data(config_ls)
            model, _ = load_model(config_ls)

            model.to(device)
            model.double()
            model.eval()

            inv_error = []
            with torch.no_grad():
                for data in train_loader:
                    X, presence, target = [d.to(device).double() for d in data]
                    angles = 2 * math.pi * rng.random(20)
                    for angle in angles:
                        outs = []
                        for a, translate in [(0, 0), (1 * angle, 0 * angle)]:
                            torch.cuda.empty_cache()
                            Xt = rotate(X, a) + translate
                            if fix_seed_before_forw_pass:
                                torch.manual_seed(0)
                            out = get_logits(
                                model,
                                [[Xt[:, :], torch.ones_like(presence[:, :])], target],
                            )
                            out = out.detach().clone()
                            outs.append(out)
                        inv_error.append((torch.abs(outs[0] - outs[1])).max().item())
                inv_errors.append(inv_error)

            print(f"Completed {idx + 1}/{len(lift_samples_p)} configs.")

        inv_errors_multiple_runs.append(inv_errors)
        print(f"Finished {idx_run + 1}/{num_runs} runs.")
    return np.array(inv_errors_multiple_runs)


# %%

if __name__ == "__main__":
    lift_samples_p = np.logspace(0, 2.4, 25, dtype=int)[::-1]
    lift_samples_p = sorted(list(set(lift_samples_p)))

    start_time = time.time()
    inv_errors = analyze_invariance(lift_samples_p, num_runs=10)
    print(f"Took {time.time() - start_time:.2f} seconds.")

    fig, ax = plt.subplots(figsize=(5 * 0.85, 5 * 0.85))
    avg_inv_errors = np.array(
        [[np.median(e) for e in inv_errs] for inv_errs in inv_errors]
    )
    avg_inv_errors_median = np.median(avg_inv_errors, axis=0)
    avg_inv_errors_median[avg_inv_errors_median < 6e-17] = 6e-17

    ax.plot(
        lift_samples_p,
        avg_inv_errors_median,
        marker="o",
        markersize=4,
        linewidth=2,
        color="dodgerblue",
    )
    ax.fill_between(
        lift_samples_p,
        np.quantile(avg_inv_errors, 0.25, axis=0),
        np.quantile(avg_inv_errors, 0.75, axis=0),
        alpha=0.2,
        color="dodgerblue",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Invariance error (in $\ell^\infty$ norm)")
    ax.set_xlabel("Number of lift samples")
    ax.grid(ls="dotted")
    fig.savefig("./diagnostics/invariance_error.pdf")
