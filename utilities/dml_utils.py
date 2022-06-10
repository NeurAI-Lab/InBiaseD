import os
import os.path
import numpy as np

def write_csv(test_accuracy_m1, test_accuracy_m2, args):

    base_name = "%s_%s_model1_%s_model2_%s_mode_%s_%sepochs" % (args.exp_identifier, args.model1_architecture, args.model2_architecture, args.mode, args.dataset, args.epochs)
    file = os.path.join(args.OUTPUT_DIR, base_name, "results.csv")

    do_write_csv(args, file, test_accuracy_m1, test_accuracy_m2)


def do_write_csv(args, file, test_accuracy_m1, test_accuracy_m2):
    names = [
        "exp_identifier"
        "model1",
        "model2",
        "mode",
        "feature_prior",
        "dataset",
        "epochs",
        "accuracy_model1",
        "accuracy_model2",
        "output_path"
    ]

    values = [
        args.exp_identifier,
        args.model1_architecture,
        args.model2_architecture,
        args.mode,
        args.ft_prior,
        args.dataset,
        args.epochs,
        test_accuracy_m1,
        test_accuracy_m2,
        os.path.dirname(file)
    ]

    np.savetxt(file, (names, values), delimiter=",", fmt="%s")