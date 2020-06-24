import os
import sys

sys.path.append(".")
from os import path as osp
import time
import torch
from torch.utils.tensorboard import SummaryWriter

import forge
from forge import flags
import forge.experiment_tools as fet
from copy import deepcopy
import deepdish as dd
from tqdm import tqdm

from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    print_reports,
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
    model = model.to(device)

    # Prepare environment
    results_folder_name = os.path.join(
        "results",
        data_name,
        model_name,
        (
            config.run_name
            + "_bs"
            + str(config.batch_size)
            + "_lr"
            + str(config.learning_rate)
        ),
    )

    config.charge_scale = float(config.charge_scale.cpu().numpy())
    config.ds_stats = [float(s.cpu().numpy()) for s in config.ds_stats]

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

    # Try to restore model and optimizer from checkpoint
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, model_opt)
    else:
        start_epoch = 1

    train_iter = (start_epoch - 1) * (
        len(dataloaders["train"].dataset) // config.batch_size
    ) + 1

    print("Starting training at epoch = {}, iter = {}".format(start_epoch, train_iter))

    # Setup tensorboard writing
    summary_writer = SummaryWriter(logdir)

    report_all = {}
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, 0.0)
    print("finished saving model at epoch 0 before training")

    # Training
    start_t = time.perf_counter()
    if config.log_train_values:
        train_emas = ExponentialMovingAverage(alpha=config.ema_alpha, debias=True)

    for epoch in tqdm(range(start_epoch, config.train_epochs + 1)):
        model.train()

        for batch_idx, data in enumerate(dataloaders["train"]):
            data = {k: v.to(device) for k, v in data.items()}
            outputs = model(data, compute_loss=True)

            model_opt.zero_grad()
            outputs.loss.backward(retain_graph=False)
            model_opt.step()

            if config.log_train_values:
                if batch_idx % config.report_loss_every == 0:
                    reports = train_emas(parse_reports(outputs.reports))
                    reports["time"] = time.perf_counter() - start_t

                    log_tensorboard(
                        summary_writer, train_iter, reports, "train_metrics/"
                    )
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
                    test_mae = 0.0
                    test_loss = 0.0
                    for data in dataloaders["test"]:
                        data = {k: v.to(device) for k, v in data.items()}
                        outputs = model(data, compute_loss=True)
                        test_mae += outputs.mae
                        test_loss += outputs.loss

                    reports = {}
                    reports["test_mae"] = test_mae / len(dataloaders["test"])
                    reports["test_loss"] = test_loss / len(dataloaders["test"])

                    reports = parse_reports(reports)

                    reports["time"] = time.perf_counter() - start_t

                    if report_all == {}:
                        report_all = deepcopy(reports)

                        for d in reports.keys():
                            report_all[d] = [report_all[d]]
                    else:
                        for d in reports.keys():
                            report_all[d].append(reports[d])

                    log_tensorboard(summary_writer, train_iter, reports, "test")
                    print_reports(
                        reports,
                        start_t,
                        epoch,
                        batch_idx,
                        len(dataloaders["train"].dataset) // config.batch_size,
                        prefix="test",
                    )

            train_iter += 1

            if train_iter % config.save_check_points == 0:
                save_checkpoint(
                    checkpoint_name, train_iter, model, model_opt, outputs.loss
                )

        with torch.no_grad():
            valid_mae = 0.0
            valid_loss = 0.0
            for data in dataloaders["valid"]:
                data = {k: v.to(device) for k, v in data.items()}
                outputs = model(data, compute_loss=True)
                valid_mae += outputs.mae
                valid_loss += outputs.loss

            reports = {}
            reports["valid_mae"] = test_mae / len(dataloaders["valid"])
            reports["valid_loss"] = test_loss / len(dataloaders["valid"])

            reports = parse_reports(reports)

            reports["time"] = time.perf_counter() - start_t

            log_tensorboard(summary_writer, train_iter, reports, "valid")
            print_reports(
                reports,
                start_t,
                epoch,
                batch_idx,
                len(dataloaders["train"].dataset) // config.batch_size,
                prefix="valid",
            )

        dd.io.save(logdir + "/results_dict.h5", report_all)

        save_checkpoint(checkpoint_name, train_iter, model, model_opt, outputs.loss)


if __name__ == "__main__":
    main()
