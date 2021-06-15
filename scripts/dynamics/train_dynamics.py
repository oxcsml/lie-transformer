from os import path as osp
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import transformers

# For reproducibility while researching, but might affect speed!
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
import deepdish as dd
from tqdm import tqdm
from types import SimpleNamespace


from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    parse_reports_cpu,
    print_reports,
    load_checkpoint,
    save_checkpoint,
    nested_to,
    param_count,
)
from attrdict import AttrDict


# %%

#####################################################################################################################
# Command line flags
#####################################################################################################################
# Directories
# flags.DEFINE_string("data_dir", "data/", "Path to data directory")
flags.DEFINE_string(
    "results_dir", "checkpoints/", "Top directory for all experimental results."
)

# Configuration files to load
flags.DEFINE_string(
    "data_config",
    "configs/dynamics/spring_dynamics_data.py",
    "Path to a data config file.",
)
flags.DEFINE_string(
    "model_config",
    "configs/dynamics/eqv_transformer_model.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string(
    "run_name",
    "test_graph",
    "Name of this job and name of results folder.",
)
flags.DEFINE_boolean("resume", False, "Tries to resume a job if True.")

# Logging
flags.DEFINE_integer(
    "report_loss_every", 10, "Number of iterations between reporting minibatch loss."
)
flags.DEFINE_integer(
    "save_check_points",
    50,
    "frequency with which to save checkpoints, in number of epoches.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_integer(
    "total_evaluations",
    100,
    "Maximum number of evaluations on test and validation data during training.",
)

# Optimization
flags.DEFINE_integer("train_epochs", 200, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 100, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-3, "Adam learning rate.")
flags.DEFINE_float("beta1", 0.9, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.999, "Adam Beta 2 parameter")
flags.DEFINE_string("lr_schedule", "cosine_annealing", "Learning rate schedule.")

# GPU device
flags.DEFINE_integer("device", 0, "GPU to use.")

# Debug mode tracks more stuff
flags.DEFINE_boolean("debug", True, "Track and show on tensorboard more metrics.")
flags.DEFINE_boolean(
    "save_test_predictions",
    True,
    "Makes and saves test predictions on one or more test sets (e.g. 5-step and 100-step predictions) at the end of training.",
)
flags.DEFINE_boolean(
    "log_val_test", True, "Turns off computation of validation and test errors."
)
#####################################################################################################################


def evaluate(model, loader, device):
    reports = None
    for data in loader:
        data = nested_to(data, device, torch.float32)
        outputs = model(data)

        if reports is None:
            reports = {k: v.detach().clone().cpu() for k, v in outputs.reports.items()}
        else:
            for k, v in outputs.reports.items():
                reports[k] += v.detach().clone().cpu()

    for k, v in reports.items():
        reports[k] = v / len(
            loader
        )  # SZ: note this can be slightly incorrect if mini-batch sizes vary (if batch_size doesn't divide train_size), but approximately correct.

    return reports


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
    dataloaders, data_name = fet.load(config.data_config, config=config)

    train_loader = dataloaders["train"]
    test_loader = dataloaders["test"]
    val_loader = dataloaders["val"]

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model = model.to(device)
    print(model)

    # Prepare environment
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
        ("batch_norm", "bn"),
        ("channel_width", "width"),
        ("attention_fn", "attfn"),
        ("output_mlp_scale", "mlpscale"),
        ("train_epochs", "epochs"),
        ("block_norm", "block"),
        ("kernel_type", "ktype"),
        ("architecture", "arch"),
        ("activation_function", "act"),
        ("space_dim", "spacedim"),
        ("num_particles", "prtcls"),
        ("n_train", "ntrain"),
        ("group", "group"),
        ("lift_samples", "ls"),
    ]

    run_name = ""  # config.run_name
    for config_param in params_in_run_name:
        attr = config_param[0]
        abbrev = config_param[1]

        if hasattr(config, attr):
            run_name += abbrev
            run_name += str(getattr(config, attr))
            run_name += "_"

    results_folder_name = osp.join(
        data_name,
        model_name,
        config.run_name,
        run_name,
    )

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Print flags
    fet.print_flags()

    # Setup optimizer
    model_params = model.predictor.parameters()

    opt_learning_rate = config.learning_rate
    model_opt = torch.optim.Adam(
        model_params, lr=opt_learning_rate, betas=(config.beta1, config.beta2)
    )

    if config.lr_schedule == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_opt, config.train_epochs
        )
    else:
        raise NotImplementedError(f"{config.lr_schedule} not implemented.")

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
    report_all_val = {}

    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, loss=0.0)
    print("finished saving model at epoch 0 before training")

    num_params = param_count(model)
    print(f"Number of model parameters: {num_params}")

    # Training
    start_t = time.time()

    total_train_iters = len(train_loader) * config.train_epochs
    iters_per_eval = max(1, int(total_train_iters / config.total_evaluations))

    assert (
        config.n_train % min(config.batch_size, config.n_train) == 0
    ), "Batch size doesn't divide dataset size. Can be inaccurate for loss computation (see below)."

    training_failed = False
    best_val_loss_so_far = 1e7

    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(train_loader):
            data = nested_to(
                data, device, torch.float32
            )  # the format is ((z0, sys_params, ts), true_zs) for data
            true_zs = data[-1]
            if config.model_with_dict:
                outputs = model(data)
            else:
                pred_zs = model(data)
                loss = (pred_zs - true_zs).pow(2).mean()
                outputs = AttrDict({"loss": loss, "prediction": pred_zs})
                outputs.reports = AttrDict({"mse": loss})

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
                    log_tensorboard(
                        summary_writer,
                        train_iter,
                        {"lr": model_opt.param_groups[0]["lr"]},
                        "hyperparams/",
                    )

            # Do learning rate schedule steps per STEP for cosine_annealing_warmup
            if config.lr_schedule == "cosine_annealing_warmup":
                scheduler.step()

            # Logging and evaluation
            if (
                train_iter % iters_per_eval == 0 or (train_iter == total_train_iters)
            ) and config.log_val_test:  # batch_idx % config.evaluate_every == 0:
                model.eval()
                with torch.no_grad():
                    reports = evaluate(model, test_loader, device)
                    reports = parse_reports(reports)
                    reports["time"] = time.time() - start_t
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

                    # repeat for validation data
                    reports = evaluate(model, val_loader, device)
                    reports = parse_reports(reports)
                    reports["time"] = time.time() - start_t
                    if report_all_val == {}:
                        report_all_val = deepcopy(reports)

                        for d in reports.keys():
                            report_all_val[d] = [report_all_val[d]]
                    else:
                        for d in reports.keys():
                            report_all_val[d].append(reports[d])

                    log_tensorboard(summary_writer, train_iter, reports, "val/")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(train_loader.dataset) // config.batch_size,
                        prefix="val",
                    )

                    if report_all_val["mse"][-1] < best_val_loss_so_far:
                        save_checkpoint(
                            checkpoint_name,
                            f"early_stop",
                            model,
                            model_opt,
                            loss=outputs.loss,
                        )
                        best_val_loss_so_far = report_all_val["mse"][-1]

                model.train()

            train_iter += 1

        # Do learning rate schedule steps per *epoch* for cosine_annealing
        if config.lr_schedule == "cosine_annealing":
            scheduler.step()

        if epoch % config.save_check_points == 0:
            save_checkpoint(
                checkpoint_name, train_iter, model, model_opt, loss=outputs.loss
            )

        dd.io.save(logdir + "/results_dict_train.h5", train_reports)
        dd.io.save(logdir + "/results_dict.h5", report_all)
        dd.io.save(logdir + "/results_dict_val.h5", report_all_val)

    # always save final model
    save_checkpoint(checkpoint_name, train_iter, model, model_opt, loss=outputs.loss)

    if config.save_test_predictions:
        print("Starting to make model predictions on test sets for *final model*.")
        for chunk_len in [5, 100]:
            start_t_preds = time.time()
            data_config = SimpleNamespace(
                **{
                    **config.__dict__["__flags"],
                    **{"chunk_len": chunk_len, "batch_size": 500},
                }
            )
            dataloaders, data_name = fet.load(config.data_config, config=data_config)
            test_loader_preds = dataloaders["test"]

            torch.cuda.empty_cache()
            with torch.no_grad():
                preds = []
                true = []
                num_datapoints = 0
                for idx, d in enumerate(test_loader_preds):
                    true.append(d[-1])
                    d = nested_to(d, device, torch.float32)
                    outputs = model(d)

                    pred_zs = outputs.prediction
                    preds.append(pred_zs)

                    num_datapoints += len(pred_zs)

                    if num_datapoints >= 2000:
                        break

                preds = torch.cat(preds, dim=0).cpu()
                true = torch.cat(true, dim=0).cpu()

                save_dir = osp.join(logdir, f"traj_preds_{chunk_len}_steps_2k_test.pt")
                torch.save(preds, save_dir)

                save_dir = osp.join(logdir, f"traj_true_{chunk_len}_steps_2k_test.pt")
                torch.save(true, save_dir)

                print(
                    f"Completed making test predictions for chunk_len = {chunk_len} in {time.time() - start_t_preds:.2f} seconds."
                )


if __name__ == "__main__":
    main()
