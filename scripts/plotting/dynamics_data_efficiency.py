# %%
import deepdish as dd
import os
import glob
import matplotlib.pyplot as plt
import json
import numpy as np


def get_test_mses(exps, early_stop=False):
    print(f"Number of experiments: {len(exps)}")

    n_trains = []
    test_mses = []

    for exp in exps:
        results_test = dd.io.load(os.path.join(exp, "results_dict.h5"))
        # results_train = dd.io.load(os.path.join(exp, "results_dict_train.h5"))

        with open(os.path.join(exp, "flags.json"), "r") as f:
            flags = json.load(f)

        n_train = flags["n_train"]
        n_trains.append(n_train)

        if early_stop:
            results_val = dd.io.load(os.path.join(exp, "results_dict_val.h5"))
            assert len(results_val['mse']) == len(results_test['mse'])
            min_idx = np.argmin(results_val['mse'])
            test_mses.append(results_test["mse"][min_idx])# - results_train[-1]['mse'])
        else:
            test_mses.append(results_test["mse"][-1])# - results_train[-1]['mse'])

        # if len(results_test["mse"]) > 100:
        #     print(len(results_test["time"]), n_train)
        #     print(flags["train_epochs"])

    test_mses_grouped = {}
    for n_train in sorted(n_trains):
        if n_train not in test_mses_grouped.keys():
            idxs = np.where(np.array(n_trains) == n_train)[0]
            test_mses_grouped[n_train] = np.array(test_mses)[idxs]

    n_trains = np.array(list(test_mses_grouped.keys()))
    test_mses_grouped = np.array(list(test_mses_grouped.values()))

    return n_trains, test_mses_grouped

# %%

# experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_2"
# exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# # exps = [e for e in exps if any(ms in e for ms in ["mseed6", "mseed7", "mseed8"])]

# n_trains, test_mses_grouped = get_test_mses(exps)

# experiment_dir = "checkpoints/spring_dynamics/HLieResNet_Dynamics/eff_n_2"
# exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# # exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

# _, lieconv_test_mses_my_run = get_test_mses(exps)


# # TODO: change back to 64 3 32
# experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_small_64_10_32" #eff_n_small_160_5_16"
# exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

# n_trains, test_mses_grouped_64_3_32 = get_test_mses(exps)

experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_small_160_5_16" #eff_n_small_160_5_16"
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_160_5_16_old = get_test_mses(exps)

experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_2_epochs400_2" #eff_n_small_160_5_16"
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_400epochs = get_test_mses(exps)

# NEW RUNS

experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_10ms" #eff_n_small_160_5_16"
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_160_5_16 = get_test_mses(exps, early_stop=True)

experiment_dir = "checkpoints/spring_dynamics/HLieResNet_Dynamics/eff_n_10ms" #eff_n_small_160_5_16"
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, lieconv_test_mses_my_run = get_test_mses(exps, early_stop=True)

# As from paper
lieconv_test_mses = [
    0.002746623150069,
    0.006711040169603,
    0.000870870963576,
    0.001954308277967,
    0.000288127224416,
    0.000136854367087,
    0.000117921646235,
    1.701950333157e-05,
    7.74721300031229e-06,
    2.30455249388161e-06,
]

# lieconv_test_mses_my_run = np.array(
#     [
#         [4.77109989e-03, 1.04175229e-03, 9.33031901e-04],
#         [1.43209705e-03, 1.05767068e-03, 1.66209433e-02],
#         [8.42396636e-04, 1.36285415e-03, 8.90521798e-04],
#         [9.31223854e-04, 4.60727053e-04, 1.07832893e-04],
#         [9.46964137e-05, 3.17952188e-04, 4.51914675e-04],
#         [9.90702538e-05, 1.08831257e-04, 3.58304715e-05],
#         [1.96911606e-05, 6.50479924e-05, 1.82115436e-05],
#         [3.65901178e-06, 2.99104158e-06, 5.84325835e-06],
#         [2.13932526e-06, 1.39843530e-06, 1.07856181e-06],
#         [1.45959191e-07, 5.70400459e-07, 2.21150032e-07],
#     ]
# )


# %%

fig, ax = plt.subplots(figsize=(6*1.1, 4*1.1))



# loop through bars and caps and set the alpha value
# [bar.set_alpha(0.5) for bar in bars]
# [cap.set_alpha(0.5) for cap in caps]

ax.plot(
    n_trains,
    lieconv_test_mses,
    label="LieConv (in paper). Params: 895K.",
    ms=5,
    # marker="v",
    color="dodgerblue",
)

# markers, caps, bars = ax.errorbar(
ax.plot(
    n_trains,
    np.median(lieconv_test_mses_my_run, axis=1),
    # yerr=[
    #     np.median(lieconv_test_mses_my_run, axis=1)
    #     - np.min(lieconv_test_mses_my_run, axis=1),
    #     np.max(lieconv_test_mses_my_run, axis=1)
    #     - np.median(lieconv_test_mses_my_run, axis=1),
    # ],
    label="LieConv (reprod. w/o ES). Params: 895K.",
    # marker="v",
    ms=5,
    # elinewidth=0.0005,
    color="dodgerblue",
    ls="--",
    # capsize=2,
    # capthick=2,
)

ax.scatter(
    np.tile(
        n_trains.reshape(len(n_trains), -1), (1, lieconv_test_mses_my_run.shape[1])
    ).flatten(),
    lieconv_test_mses_my_run.flatten(),
    color="dodgerblue",
    s=5,
    alpha=0.5,
    ec=None,
)

# markers, caps, bars = ax.errorbar(
# ax.plot(
#     n_trains,
#     np.median(test_mses_grouped, axis=1),
#     # yerr=[
#     #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
#     #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
#     # ],
#     label="Equiv. Transf. Params: 1.7M.",
#     # marker="v",
#     ms=5,
#     # elinewidth=0.0005,
#     color="crimson",
#     # capsize=2,
#     # capthick=2,
# )#

# ax.scatter(
#     np.tile(
#         n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped.shape[1])
#     ).flatten(),
#     test_mses_grouped.flatten(),
#     color="crimson",
#     s=5,
#     ec=None,
#     alpha=0.5,
# )


# ax.plot(
#     n_trains,
#     np.median(test_mses_grouped_400epochs, axis=1),
#     # yerr=[
#     #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
#     #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
#     # ],
#     label="Equiv. Transf (epochs x 2). Params: 1.7M.",
#     # marker="v",
#     ms=5,
#     # elinewidth=0.0005,
#     color="darkorange",
#     # capsize=2,
#     # capthick=2,
# )#

# ax.scatter(
#     np.tile(
#         n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped_400epochs.shape[1])
#     ).flatten(),
#     test_mses_grouped_400epochs.flatten(),
#     color="darkorange",
#     s=5,
#     ec=None,
#     alpha=0.5,
# )


ax.plot(
    n_trains,
    np.median(test_mses_grouped_160_5_16, axis=1),
    # yerr=[
    #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
    #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
    # ],
    label="Equiv. Transf. Params: 842K.",
    # marker="v",
    ms=5,
    # elinewidth=0.0005,
    color="crimson",
    # capsize=2,
    # capthick=2,
)

ax.scatter(
    np.tile(
        n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped_160_5_16.shape[1])
    ).flatten(),
    test_mses_grouped_160_5_16.flatten(),
    color="crimson",
    s=5,
    alpha=0.5,
    ec=None,
)

# ax.plot(
#     n_trains,
#     np.median(test_mses_grouped_160_5_16_old, axis=1),
#     # yerr=[
#     #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
#     #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
#     # ],
#     label="Equiv. Transf. Params: 842K.",
#     # marker="v",
#     ms=5,
#     # elinewidth=0.0005,
#     color="crimson",
#     # capsize=2,
#     # capthick=2,
# )

# ax.scatter(
#     np.tile(
#         n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped_160_5_16_old.shape[1])
#     ).flatten(),
#     test_mses_grouped_160_5_16_old.flatten(),
#     color="crimson",
#     s=5,
#     alpha=0.5,
#     ec=None,
# )

# ax.plot(
#     n_trains,
#     np.median(test_mses_grouped_64_3_32, axis=1),
#     # yerr=[
#     #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
#     #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
#     # ],
#     label="Equiv. Transf. Params: 112K.",
#     # marker="v",
#     ms=5,
#     # elinewidth=0.0005,
#     color="black",
#     # capsize=2,
#     # capthick=2,
# )

# ax.scatter(
#     np.tile(
#         n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped_64_3_32.shape[1])
#     ).flatten(),
#     test_mses_grouped_64_3_32.flatten(),
#     color="black",
#     s=5,
#     alpha=0.5,
#     ec=None,
# )

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel("Test MSE")
ax.set_xlabel("Training data size")
ax.legend(fontsize=8)

ax.set_title("Spring dynamics data efficiency")

fig.savefig("./sheh_scripts/plots/spring_dynamics_data_efficiency_more_epochs.pdf")

# %%

# ---------------------------------------------------------------------------- #
#                                 n-body curves                                #
# ---------------------------------------------------------------------------- #


experiment_dir = "checkpoints/nbody_dynamics/EqvTransformer_Dynamics/eff_n_3body_160_8heads" #160_8, 360_4
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_160_8 = get_test_mses(exps, early_stop=True)

experiment_dir = "checkpoints/nbody_dynamics/EqvTransformer_Dynamics/eff_n_3body_360_4heads" #160_8, 360_4
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_360_4 = get_test_mses(exps, early_stop=True)

experiment_dir = "checkpoints/nbody_dynamics/EqvTransformer_Dynamics/eff_n_3body_256_16heads" #160_8, 360_4
exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

n_trains, test_mses_grouped_256_16 = get_test_mses(exps, early_stop=True)

experiment_dir = "checkpoints/nbody_dynamics/HLieResNet_Dynamics/eff_n_3body"
exps = glob.glob(os.path.join(experiment_dir, "*/1"))
# exps = [e for e in exps if any(ms in e for ms in ["mseed3", "mseed4", "mseed5"])]

_, lieconv_test_mses_my_run = get_test_mses(exps, early_stop=True)


# experiment_dir = "checkpoints/spring_dynamics/EqvTransformer_Dynamics/eff_n_small_128_4_32" #eff_n_small_160_5_16"
# exps = glob.glob(os.path.join(experiment_dir, "*/*"))
# # exps = [e for e in exps if any(ms in e for ms in ["mseed6", "mseed7", "mseed8"])]

# n_trains, test_mses_grouped_new = get_test_mses(exps)

# %%

fig, ax = plt.subplots(figsize=(6*1.1, 4*1.1))

# ax.set_ylim(1e-4, 1e2)

for data_plot, c, l in zip([test_mses_grouped_160_8, test_mses_grouped_256_16, test_mses_grouped_360_4], ["crimson", "black", "darkorange"], ["(dim: 160. heads: 8)", "(dim: 256. heads: 16)", "(dim: 360. heads: 4)"]):
    # markers, caps, bars = ax.errorbar(
    ax.plot(
        n_trains,
        np.median(data_plot, axis=1),
        # yerr=[
        #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
        #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
        # ],
        label="Equiv. Transf. " + l,
        # marker="v",
        ms=5,
        # elinewidth=0.0005,
        color=c,
        # capsize=2,
        # capthick=2,
    )#

    ax.scatter(
        np.tile(
            n_trains.reshape(len(n_trains), -1), (1, data_plot.shape[1])
        ).flatten(),
        data_plot.flatten(),
        color=c,
        s=5,
        ec=None,
        alpha=0.5,
    )


# markers, caps, bars = ax.errorbar(
ax.plot(
    n_trains,
    np.median(lieconv_test_mses_my_run, axis=1),
    # yerr=[
    #     np.median(lieconv_test_mses_my_run, axis=1)
    #     - np.min(lieconv_test_mses_my_run, axis=1),
    #     np.max(lieconv_test_mses_my_run, axis=1)
    #     - np.median(lieconv_test_mses_my_run, axis=1),
    # ],
    label="LieConv (reproduced w/o early stopping)",
    # marker="v",
    ms=5,
    # elinewidth=0.0005,
    color="dodgerblue",
    # ls="--",
    # capsize=2,
    # capthick=2,
)

ax.scatter(
    np.tile(
        n_trains.reshape(len(n_trains), -1), (1, lieconv_test_mses_my_run.shape[1])
    ).flatten(),
    lieconv_test_mses_my_run.flatten(),
    color="dodgerblue",
    s=5,
    alpha=0.5,
    ec=None,
)


# markers, caps, bars = ax.errorbar(
# ax.plot(
#     n_trains,
#     np.median(test_mses_grouped_new, axis=1),
#     # yerr=[
#     #     np.median(test_mses_grouped, axis=1) - np.min(test_mses_grouped, axis=1),
#     #     np.max(test_mses_grouped, axis=1) - np.median(test_mses_grouped, axis=1),
#     # ],
#     label="Equiv. Transf.",
#     # marker="v",
#     ms=5,
#     # elinewidth=0.0005,
#     color="black",
#     # capsize=2,
#     # capthick=2,
# )

# ax.scatter(
#     np.tile(
#         n_trains.reshape(len(n_trains), -1), (1, test_mses_grouped_new.shape[1])
#     ).flatten(),
#     test_mses_grouped_new.flatten(),
#     color="black",
#     s=5,
#     alpha=0.5,
#     ec=None,
# )


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel("Test MSE")
ax.set_xlabel("Training data size")
ax.legend(fontsize=8)

ax.set_title("n-body dynamics data efficiency")

fig.savefig("./sheh_scripts/plots/spring_dynamics_data_efficiency.pdf")

# %%







