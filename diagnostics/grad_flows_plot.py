from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import deepdish as dd
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams.update({"figure.autolayout": True})

grad_flows_lc = dd.io.load(
    "checkpoints/spring_dynamics/HLieResNet_Dynamics/test_grad_flow_bs100_lr0_001_nlayers4_mseed0_lrschedcosine_annealing_k384/1/grad_flows.h5"
)
grad_flows_et = dd.io.load(
    "checkpoints/spring_dynamics/EqvTransformer_Dynamics/test_grad_flow_bs100_lr0_001_nheads8_nlayers4_hdim256_kdim16_locattTrue_mseed0_lrschedcosine_annealing_lnTrue_bnattFalse_bnFalse/2/grad_flows.h5"
)

# %%

assert len(grad_flows_et) == len(grad_flows_lc)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 9))


def animate(i):
    for ax in axes:
        ax.clear()
    for itr, (ax, grad_flows) in enumerate(zip(axes, [grad_flows_et, grad_flows_lc])):
        # ax = axes[0]

        layers = grad_flows[i]["layers"]
        layers = [s[10:] for s in layers]
        title = "Eqv Transf" if itr == 0 else "Lie Conv"

        ax.set_xticks(range(0, len(layers)))
        xtickNames = ax.set_xticklabels(layers)
        plt.setp(xtickNames, rotation=90)

        ax.set_xlim(-1, len(layers))
        ax.set_ylim(1e-12, 1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient")
        ax.set_title(title + ": Gradient flow" + f" (train itr: {i})")
        # ax.grid(True)
        ax.legend(
            [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4)],
            ["max-gradient", "mean-gradient", "zero-gradient"],
            loc="upper right",
        )

        ave_grads = grad_flows[i]["ave_grads"]
        max_grads = grad_flows[i]["max_grads"]

        x = range(len(layers))

        ax.bar(x, max_grads, lw=1, color="c")
        bar = ax.bar(x, ave_grads, lw=1, color="b")

        # fig.suptitle(f'Training iteration: {i}')
        # fig.tight_layout(rect=[0, 0.03, 1., 0.95])

        plt.tight_layout()

        print(i)

    return bar


anim = FuncAnimation(
    fig, animate, repeat=False, blit=True, interval=150, frames=range(0, 6000, 10)
)

anim.save("./sheh_scripts/plots/grad_flows_finers.gif", writer="imagemagick")
