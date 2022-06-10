import os
import csv
import numpy as np

def print_result(file):
    f = open(file)  # encoding="utf8"
    reader = csv.DictReader(f, delimiter=",")
    data = [row for row in reader]
    print(data)


def save_results_normal(args, file, test_loss, test_accuracy, best_epoch, best_accuracy,
                        seed=0, mu=0, sigma = 0):

    names = [
        "exp",
        "mode",
        "seed",
        "lr",
        "epochs",
        "dataset",
        "network",
        "feature_prior",
        "test_loss",
        "test_acc",
        "best_epoch",
        "best_test_acc",
        "mu",
        "sigma",
        "model_dir",
    ]

    values = [
        args.exp_identifier,
        args.mode,
        seed,
        args.lr,
        args.epochs,
        args.dataset,
        args.model_architecture,
        args.ft_prior,
        test_loss,
        test_accuracy * 100,
        best_epoch,
        best_accuracy * 100,
        mu,
        sigma,
        os.path.join(args.experiment_name, 'checkpoints'),
    ]

    if os.path.isfile(file):
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        np.savetxt(file, (names, values), delimiter=",", fmt="%s")

def save_results_inbiased(args, file, test_loss_m1, test_accuracy_m1, best_epoch_m1, best_accuracy_m1,
                        test_loss_m2, test_accuracy_m2, seed=0, mu=0, sigma = 0):

    names = [
        "exp",
        "mode",
        "seed",
        "lr",
        "epochs",
        "dataset",
        "network",
        "feature_prior",
        "test_loss_m1",
        "test_acc_m1",
        "best_epoch_m1",
        "best_test_acc_m1",
        "test_loss_m2",
        "test_acc_m2",
        "mu",
        "sigma",
        "model_dir",
    ]

    values = [
        args.exp_identifier,
        args.mode,
        seed,
        args.lr,
        args.epochs,
        args.dataset,
        args.model1_architecture,
        args.ft_prior,
        test_loss_m1,
        test_accuracy_m1 * 100,
        best_epoch_m1,
        best_accuracy_m1 * 100,
        test_loss_m2,
        test_accuracy_m2 * 100,
        mu,
        sigma,
        os.path.join(args.experiment_name, 'checkpoints'),
    ]

    if os.path.isfile(file):
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        np.savetxt(file, (names, values), delimiter=",", fmt="%s")

def save_results_adv(args, file, test_loss_m1, test_accuracy_m1, adv_accuracy_m1,
                        test_loss_m2, test_accuracy_m2, seed=0, mu=0, sigma = 0):

    names = [
        "exp",
        "mode",
        "seed",
        "lr",
        "epochs",
        "dataset",
        "network",
        "feature_prior",
        "test_loss_m1",
        "test_acc_m1",
        "adv_test_acc_m1",
        "test_loss_m2",
        "test_acc_m2",
        "mu",
        "sigma",
        "adv_loss_type",
        "epsilon",
        "model_dir",
    ]

    values = [
        args.exp_identifier,
        args.mode,
        seed,
        args.lr,
        args.epochs,
        args.dataset,
        args.model1_architecture,
        args.ft_prior,
        test_loss_m1,
        test_accuracy_m1 * 100,
        adv_accuracy_m1 * 100,
        test_loss_m2,
        test_accuracy_m2 * 100,
        mu,
        sigma,
        args.adv_loss_type,
        args.epsilon,
        os.path.join(args.experiment_name, 'checkpoints'),
    ]

    if os.path.isfile(file):
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        np.savetxt(file, (names, values), delimiter=",", fmt="%s")