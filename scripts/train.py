from os import path as osp
import time
import torch
from torch.utils.tensorboard import SummaryWriter

# For reproducibility while researching, but might affect speed!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
from tqdm import tqdm
import deepdish as dd
from itertools import chain


from eqv_transformer.train_tools import (
    rotate,
    log_tensorboard,
    parse_reports,
    parse_reports_cpu,
    print_reports,
    load_checkpoint,
    save_checkpoint,
    ExponentialMovingAverage,
    param_count,
    get_component,
    get_average_norm,
)


# %%

#####################################################################################################################
# Command line flags
#####################################################################################################################
# Directories
flags.DEFINE_string("data_dir", "data/", "Path to data directory")
flags.DEFINE_string(
    "results_dir", "checkpoints/", "Top directory for all experimental results."
)

# Configuration files to load
flags.DEFINE_string(
    "data_config",
    "configs/constellation/constellation.py",
    "Path to a data config file.",
)
flags.DEFINE_string(
    "model_config",
    "configs/constellation/eqv_transformer_model.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string("run_name", "main", "Name of this job and name of results folder.")
flags.DEFINE_boolean("resume", False, "Tries to resume a job if True.")

# Logging
flags.DEFINE_integer(
    "report_loss_every", 500, "Number of iterations between reporting minibatch loss."
)
flags.DEFINE_integer(
    "evaluate_every", 10000, "Number of iterations between reporting validation loss."
)
flags.DEFINE_integer(
    "save_check_points",
    50,
    "frequency with which to save checkpoints, in number of epoches.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_float(
    "ema_alpha", 0.99, "Alpha coefficient for exponential moving average of train logs."
)

# Optimization
flags.DEFINE_integer("train_epochs", 200, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 90, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-5, "SGD learning rate.")
flags.DEFINE_float("beta1", 0.5, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.9, "Adam Beta 2 parameter")

# GPU device
flags.DEFINE_integer("device", 0, "GPU to use.")

# Debug mode tracks more stuff
flags.DEFINE_boolean("debug", True, "Track and show on tensorboard more metrics.")

#####################################################################################################################


# @forge.debug_on(Exception)
def main():
    # Parse flags
    config = forge.config()

    # Set device
    if torch.cuda.is_available():
        device = f"cuda:{config.device}"
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    # Load data
    train_loader, test_loader, data_name = fet.load(config.data_config, config=config)

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model = model.to(device)
    print(model)

    # Prepare environment

    if 'set_transformer' in config.model_config:
            params_in_run_name = [
            ("batch_size", "bs"),
            ("learning_rate", "lr"),
            ("num_heads", "nheads"),
            ("patterns_reps", "reps"),
            ("train_epochs", "epochs"),
            
            ]

    else:
        params_in_run_name = [
            ("batch_size", "bs"),
            ("learning_rate", "lr"),
            ("num_heads", "nheads"),
            ("num_layers", "nlayers"),
            ("dim_hidden", "hdim"),
            ("kernel_dim", "kdim"), 
            ("location_attention", "locatt"),
            ("model_seed", "mseed"),
            ("lr_schedule", "lrsched"),
            ("layer_norm", "ln"),
            ("batch_norm_att", "bnatt"),
            ("batch_norm", "bn"),
            ("batch_norm_final_mlp", "bnfinalmlp"),
            ("k", "k"),
            ("attention_fn", "attfn"),
            ("output_mlp_scale", "mlpscale"),
            ("train_epochs", "epochs"),
            ("block_norm", "block"),
            ("kernel_type", "ktype"),
            ("architecture", "arch"),
            ("kernel_act", "actv"),
            ("patterns_reps", "reps"),
            ("lift_samples", "nsamples"),
            ("content_type", "content"),
        ]

    run_name = ""#config.run_name
    for config_param in params_in_run_name:
        attr = config_param[0]
        abbrev = config_param[1]

        if hasattr(config, attr):
            run_name += (abbrev)
            run_name += str(getattr(config, attr))
            run_name += "_"


    results_folder_name = osp.join(
        data_name,
        model_name,
        config.run_name,
        run_name,
    )

    # results_folder_name = osp.join(data_name, model_name, run_name,)

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Print flags
    fet.print_flags()

    # Setup optimizer
    model_params = model.encoder.parameters()

    opt_learning_rate = config.learning_rate
    model_opt = torch.optim.Adam(
        model_params, lr=opt_learning_rate, betas=(config.beta1, config.beta2)
    )

    # Try to restore model and optimizer from checkpoint
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, model_opt)
    else:
        start_epoch = 1

    train_iter = (start_epoch - 1) * (
        len(train_loader.dataset) // config.batch_size
    ) + 1

    print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    train_reports = []
    report_all = {}
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, loss=0.0)
    print("finished saving model at epoch 0 before training")

    if config.debug and config.model_config == 'configs/constellation/eqv_transformer_model.py': 
        model_components = [(0, [], "embedding_layer")] + list(
            chain.from_iterable(
                (
                    (k, [], f"ema_{k}"),
                    (k, ["ema", "kernel", "location_kernel"], f"ema_{k}_location_kernel"),
                    (k, ["ema", "kernel", "feature_kernel"], f"ema_{k}_feature_kernel"),
                )
                for k in range(1, config.num_layers + 1)
            )
        ) + [(config.num_layers + 2, [], 'output_mlp')] # components to track for debugging 

    num_params = param_count(model)
    print(f"Number of model parameters: {num_params}")

    # Training
    start_t = time.time()

    grad_flows = []
    training_failed = False
    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(train_loader):
            data, presence, target = [d.to(device) for d in data]
            outputs = model([data, presence], target)

            if torch.isnan(outputs.loss):
                if not training_failed:
                    epoch_of_nan = epoch
                if (epoch > epoch_of_nan + 1) and training_failed:
                    raise ValueError("Loss Nan-ed.")
                training_failed = True

            model_opt.zero_grad()
            outputs.loss.backward(retain_graph=False)
            model_opt.step()

            train_reports.append(parse_reports_cpu(outputs.reports))

            if config.log_train_values:
                reports = parse_reports(outputs.reports)
                if batch_idx % config.report_loss_every == 0:
                    log_tensorboard(summary_writer, train_iter, reports, "train/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="train",
                    )

            # Track stuff for debugging
            if config.debug:
                model.eval()
                with torch.no_grad():
                    curr_params = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }

                    # updates norm
                    if train_iter == 1:
                        update_norm = 0
                    else:
                        update_norms = []
                        for (k_prev, v_prev), (k_curr, v_curr) in zip(
                            prev_params.items(), curr_params.items()
                        ):
                            assert k_prev == k_curr
                            if ('tracked' not in k_prev): # ignore batch norm tracking. TODO: should this be ignored? if not, fix!
                                update_norms.append((v_curr - v_prev).norm(1).item()) 
                        update_norm = sum(update_norms)

                    # gradient norm
                    grad_norm = 0
                    for p in model.parameters():
                        try:
                            grad_norm += p.grad.norm(1)
                        except AttributeError:
                            pass

                    # # weights norm
                    # if config.model_config == 'configs/constellation/eqv_transformer_model.py':
                    #     model_norms = {}
                    #     for comp_name in model_components:
                    #         comp = get_component(model.encoder.net, comp_name)
                    #         norm = get_average_norm(comp)
                    #         model_norms[comp_name[2]] = norm

                    #     log_tensorboard(
                    #         summary_writer,
                    #         train_iter,
                    #         model_norms,
                    #         "debug/avg_model_norms/",
                    #     )

                    log_tensorboard(
                        summary_writer,
                        train_iter,
                        {
                            "avg_update_norm1": update_norm / num_params,
                            "avg_grad_norm1": grad_norm / num_params,
                        },
                        "debug/",
                    )
                    prev_params = curr_params

                    # # gradient flow
                    # ave_grads = []
                    # max_grads= []
                    # layers = []
                    # for n, p in model.named_parameters():
                    #     if (p.requires_grad) and (p.grad is not None): # and ("bias" not in n):
                    #         layers.append(n)
                    #         ave_grads.append(p.grad.abs().mean().item())
                    #         max_grads.append(p.grad.abs().max().item())

                    # grad_flow = {"layers": layers, "ave_grads": ave_grads, "max_grads": max_grads}
                    # grad_flows.append(grad_flow)

                model.train()

            # Logging
            if batch_idx % config.evaluate_every == 0:
                model.eval()
                with torch.no_grad():
                    reports = None
                    for data in test_loader:
                        data, presence, target = [d.to(device) for d in data]
                        # if config.data_config == "configs/constellation/constellation.py":
                            # if config.global_rotation_angle != 0.0:
                                # data = rotate(data, config.global_rotation_angle)
                        outputs = model([data, presence], target)
                        
                        if reports is None:
                            reports = {k: v.detach().clone().cpu() for k, v in outputs.reports.items()}
                        else:
                            for k, v in outputs.reports.items():
                                reports[k] += v.detach().clone().cpu()

                    for k, v in reports.items():
                        reports[k] = v / len(test_loader) # SZ: note this is slightly incorrect since mini-batch sizes can vary (if batch_size doesn't divide train_size), but approximately correct.

                    reports = parse_reports(reports)
                    reports['time'] = time.time() - start_t
                    if report_all == {}:
                        report_all = deepcopy(reports)

                        for d in reports.keys():
                            report_all[d] = [report_all[d]]
                    else:
                        for d in reports.keys():
                            report_all[d].append(reports[d])

                    log_tensorboard(summary_writer, train_iter, reports, "test/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="test",
                    )

                model.train()

            train_iter += 1

        if epoch % config.save_check_points == 0:
            save_checkpoint(
                checkpoint_name, train_iter, model, model_opt, loss=outputs.loss
            )

        dd.io.save(logdir + '/results_dict_train.h5', train_reports)
        dd.io.save(logdir + "/results_dict.h5", report_all)

        # if config.debug:
        #     # dd.io.save(logdir + "/grad_flows.h5", grad_flows)

    save_checkpoint(
        checkpoint_name, train_iter, model, model_opt, loss=outputs.loss
    )


if __name__ == "__main__":
    main()


# %%
