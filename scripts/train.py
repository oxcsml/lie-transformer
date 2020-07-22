import sys

sys.path.append("forge")
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

from eqv_transformer.train_tools import (
    log_tensorboard,
    parse_reports,
    print_reports,
    load_checkpoint,
    save_checkpoint,
    ExponentialMovingAverage,
)


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
    "data_config", "configs/constellation.py", "Path to a data config file."
)
flags.DEFINE_string(
    "model_config", "configs/eqv_transformer_model.py", "Path to a model config file."
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

    # Prepare environment
    results_folder_name = (
        config.run_name
        + "_bs"
        + str(config.batch_size)
        + "_lr"
        + str(config.learning_rate)
    )

    logdir = osp.join(config.results_dir, results_folder_name.replace(".", "_"))
    logdir, resume_checkpoint = fet.init_checkpoint(
        logdir, config.data_config, config.model_config, config.resume
    )

    checkpoint_name = osp.join(logdir, "model.ckpt")

    # Load data
    train_loader, test_loader = fet.load(config.data_config, config=config)

    # Load model
    model = fet.load(config.model_config, config).to(device)

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

    report_all = {}
    # Saving model at epoch 0 before training
    print("saving model at epoch 0 before training ... ")
    save_checkpoint(checkpoint_name, 0, model, model_opt, 0.0)
    print("finished saving model at epoch 0 before training")

    # Training
    start_t = time.perf_counter()
    if config.log_train_values:
        train_emas = ExponentialMovingAverage(alpha=config.ema_alpha, debias=True)

    for epoch in range(start_epoch, config.train_epochs + 1):
        model.train()

        for batch_idx, data in enumerate(train_loader):
            data, presence, target = [d.to(device) for d in data]
            outputs = model([data, presence], target)

            model_opt.zero_grad()
            outputs.loss.backward(retain_graph=False)
            model_opt.step()

            if config.log_train_values:
                reports = train_emas(parse_reports(outputs.reports))
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

            # Logging
            if batch_idx % config.evaluate_every == 0:
<<<<<<< HEAD
                test_acc = 0.0
=======
                reports = None
>>>>>>> 7de5220c9d33f4987e5b096da01549c2a240af33
                for data in test_loader:
                    data, presence, target = [d.to(device) for d in data]
                    outputs = model([data, presence], target)
                    outputs.reports.acc = outputs.acc

<<<<<<< HEAD
                outputs["reports"].cls_acc = test_acc / len(test_loader)

                reports = parse_reports(outputs.reports)
                reports["time"] = time.perf_counter() - start_t
=======
                    if reports is None:
                        reports = {k: v.detach().clone().cpu() for k, v in outputs.reports.items()}
                    else:
                        for k, v in outputs.reports.items():
                            reports[k] += v.detach().clone().cpu()

                for k, v in reports.items():
                    reports[k] = v / len(test_loader)

                reports = parse_reports(reports)
                reports['time'] = time.perf_counter() - start_t
>>>>>>> 7de5220c9d33f4987e5b096da01549c2a240af33
                if report_all == {}:
                    report_all = deepcopy(reports)

                    for d in reports.keys():
                        report_all[d] = [report_all[d]]
                else:
                    for d in reports.keys():
                        report_all[d].append(reports[d])

                log_tensorboard(summary_writer, train_iter, reports)
                print_reports(
                    reports,
                    start_t,
                    epoch,
                    batch_idx,
                    len(train_loader.dataset) // config.batch_size,
                    prefix="test",
                )

            train_iter += 1

            if train_iter % config.save_check_points == 0:
                save_checkpoint(
                    checkpoint_name, train_iter, model, model_opt, outputs.loss
                )

        dd.io.save(logdir + "/results_dict.h5", report_all)

        save_checkpoint(checkpoint_name, train_iter, model, model_opt, outputs.loss)


if __name__ == "__main__":
    main()
