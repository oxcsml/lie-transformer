from os import path as osp
import time
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from oil.utils.utils import cosLr

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
from collections import defaultdict
import deepdish as dd
from tqdm import tqdm

from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    print_reports,
    log_reports,
    load_checkpoint,
    save_checkpoint,
    ExponentialMovingAverage,
)

from eqv_transformer.molecule_predictor import MoleculePredictor


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
    "data_config", "configs/molecule/qm9_data.py", "Path to a data config file."
)
flags.DEFINE_string(
    "model_config",
    "configs/molecule/set_transformer.py",
    "Path to a model config file.",
)
# Job management
flags.DEFINE_string("run_name", "test", "Name of this job and name of results folder.")
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
    100000,
    "frequency with which to save checkpoints, in number of minibatches.",
)
flags.DEFINE_boolean("log_train_values", True, "Logs train values if True.")
flags.DEFINE_float(
    "ema_alpha", 0.99, "Alpha coefficient for exponential moving average of train logs."
)

# Optimization
flags.DEFINE_integer("train_epochs", 200, "Maximum number of training epochs.")
flags.DEFINE_integer("batch_size", 5, "Mini-batch size.")
flags.DEFINE_float("learning_rate", 1e-5, "SGD learning rate.")
flags.DEFINE_float("beta1", 0.5, "Adam Beta 1 parameter")
flags.DEFINE_float("beta2", 0.9, "Adam Beta 2 parameter")
flags.DEFINE_string(
    "lr_schedule",
    "cosine_warmup",
    "What learning rate schedule to use. Options: cosine, none",
)


#####################################################################################################################


# @forge.debug_on(Exception)
def main():
    # Parse flags
    config = forge.config()

    # Load data
    dataloaders, num_species, charge_scale, ds_stats, data_name = fet.load(
        config.data_config, config=config
    )

    config.num_species = num_species
    config.charge_scale = charge_scale
    config.ds_stats = ds_stats

    # Load model
    model, model_name = fet.load(config.model_config, config)
    model.to(device)

    config.charge_scale = float(config.charge_scale.numpy())
    config.ds_stats = [float(stat.numpy()) for stat in config.ds_stats]

    # Prepare environment
    run_name = (
        config.run_name
        + "_bs"
        + str(config.batch_size)
        + "_lr"
        + str(config.learning_rate)
    )

    if config.batch_fit != 0:
        run_name += "_bf" + str(config.batch_fit)

    if config.lr_schedule != "none":
        run_name += "_" + config.lr_schedule

    results_folder_name = osp.join(data_name, model_name, run_name,)

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
    # model_opt = torch.optim.SGD(model_params, lr=opt_learning_rate)

    # Cosine annealing learning rate
    if config.lr_schedule == "cosine_warmup":
        cos = cosLr(config.train_epochs)
        lr_sched = lambda e: min(e / (0.01 * config.train_epochs), 1) * cos(e)
    elif config.lr_schedule == "none":
        lr_sched = lambda e: 1.0
    else:
        raise ValueError(
            f"{config.lr_schedule} is not a recognised learning rate schedule"
        )

    lr_schedule = optim.lr_scheduler.LambdaLR(model_opt, lr_sched)

    # Try to restore model and optimizer from checkpoint
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, model_opt, lr_schedule)
    else:
        start_epoch = 1

    train_iter = (start_epoch - 1) * (
        len(dataloaders["train"].dataset) // config.batch_size
    ) + 1

    print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    report_all = defaultdict(list)
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, lr_schedule, 0.0)
    print("finished saving model at epoch 0 before training")

    # Training
    start_t = time.perf_counter()
    if config.log_train_values:
        train_emas = ExponentialMovingAverage(alpha=config.ema_alpha, debias=True)

    iters_per_epoch = len(dataloaders["train"])
    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(dataloaders["train"]):
            data = {k: v.to(device) for k, v in data.items()}

            model_opt.zero_grad()
            outputs = model(data, compute_loss=True)
            outputs.loss.backward()
            model_opt.step()

            if config.log_train_values:
                reports = parse_reports(outputs.reports)
                if batch_idx % config.report_loss_every == 0:
                    log_tensorboard(summary_writer, train_iter, reports, "train/")
                    report_all = log_reports(report_all, train_iter, reports, "train")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(dataloaders["train"].dataset) // config.batch_size,
                        prefix="train",
                    )

            # Logging
            if batch_idx % config.evaluate_every == 0:
                with torch.no_grad():
                    valid_mae = 0.0
                    for data in dataloaders["valid"]:
                        data = {k: v.to(device) for k, v in data.items()}
                        outputs = model(data, compute_loss=True)
                        valid_mae = valid_mae + outputs.mae

                outputs["reports"].valid_mae = valid_mae / len(dataloaders["valid"])

                reports = parse_reports(outputs.reports)

                log_tensorboard(summary_writer, train_iter, reports, "valid")
                report_all = log_reports(report_all, train_iter, reports, "valid")
                print_reports(
                    reports,
                    start_t,
                    epoch,
                    batch_idx,
                    len(dataloaders["train"].dataset) // config.batch_size,
                    prefix="valid",
                )

            train_iter += 1

            # Step the LR schedule
            lr_schedule.step(train_iter / iters_per_epoch)

            if train_iter % config.save_check_points == 0:
                save_checkpoint(
                    checkpoint_name,
                    train_iter,
                    model,
                    model_opt,
                    lr_schedule,
                    outputs.loss,
                )

        # Test model at end of batch
        with torch.no_grad():
            test_mae = 0.0
            for data in dataloaders["test"]:
                data = {k: v.to(device) for k, v in data.items()}
                outputs = model(data, compute_loss=True)
                test_mae = test_mae + outputs.mae

        outputs["reports"].test_mae = test_mae / len(dataloaders["test"])

        reports = parse_reports(outputs.reports)

        log_tensorboard(summary_writer, train_iter, reports, "test")
        report_all = log_reports(report_all, train_iter, reports, "test")

        print_reports(
            reports,
            start_t,
            epoch,
            batch_idx,
            len(dataloaders["train"].dataset) // config.batch_size,
            prefix="test",
        )

        reports = {"lr": lr_schedule.get_lr()[0], "time": time.perf_counter() - start_t}

        log_tensorboard(summary_writer, train_iter, reports, "stats")
        report_all = log_reports(report_all, train_iter, reports, "stats")

        # Save the reports
        dd.io.save(logdir + "/results_dict.h5", report_all)

        # Save a checkpoint
        save_checkpoint(
            checkpoint_name, train_iter, model, model_opt, lr_schedule, outputs.loss
        )


if __name__ == "__main__":
    main()
