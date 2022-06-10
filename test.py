import argparse
import os
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utilities.utils import validate
from kd_lib import utilities as utils
from models.selector import select_model
from data.transforms import build_transforms
from data.dataset import DATASETS

parser = argparse.ArgumentParser(description='Evaluation Script for classification')
# Model options
parser.add_argument('--exp_identifier', type=str, default='')
parser.add_argument('--model_architecture', type=str, default='ResNet18')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--resnet_multiplier', default=1, type=int)
parser.add_argument('--seeds', nargs='*', type=int, default=[0, 10])
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run for training the network')
# Dataset
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--cache-dataset", action="store_true", default=False)
parser.add_argument('--tiny_imagenet_path', type=str, default='/data/input/datasets/tiny_imagenet/tiny-imagenet-200')
# Device options
parser.add_argument('--cuda', action='store_true')
# evaluation options
parser.add_argument("--model_path",  default='', type=str)
parser.add_argument('--batch_size', default=1, type=int)
# storage options
parser.add_argument('--output_dir', default='experiments', type=str)
parser.add_argument("--results_csv_dir", default="None", type=str, help="specify a directory for results csv")
parser.add_argument("--append_csv", action='store_true', help="If append to existing csv")
# Feature prior options
#parser.add_argument("--train_ftprior", action="store_true", default=False)
parser.add_argument('--mode', default='normal', choices=['normal', 'feat_prior'])
parser.add_argument('--ft_prior', type=str, default='std', choices=['std', 'sobel'])
# Sobel filter options
parser.add_argument('--sobel_gauss_ksize', default=3, type=int)
parser.add_argument('--sobel_ksize', default=3, type=int)
parser.add_argument("--sobel_upsample", type=str, default='True')

parser.add_argument('--norm_std', type=str, default='False')
parser.add_argument('--norm_fp', type=str, default='False')

# ======================================================================================
# Helper Function
# ======================================================================================
def results_csv(args):

    if not args.append_csv:
        file = os.path.join(args.output_dir, "results1.csv")
        if os.path.exists(file):
            os.remove(file)
    else:
        file = os.path.join(args.output_dir, "results1.csv")
    return file

# ======================================================================================
# Training Function
# ======================================================================================
def solver(args):

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    # load dataset
    if args.dataset =='imagenet' or args.dataset =='imagenet200':
        ds_class = DATASETS[args.dataset](args.data_path, cache_dataset=True)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)
    _, transform_test = build_transforms(args, ds_class)
    testset = ds_class.get_dataset('test', None, transform_test)

    # Load model
    cifar_resnet = True
    if args.dataset == 'imagenet200':
        cifar_resnet = False
    model = select_model(args.model_architecture, args.num_classes, cifar_resnet).to(device)

    checkpoint = torch.load(args.model_path)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        model =  torch.load(args.model_path)

    # get final test accuracy
    #test_loss, test_accuracy, correct = utils.eval(model, device, test_loader)

    test_accuracy = validate(args, model, testset, use_cuda)
    print("Test: Accuracy: ({}%)".format(test_accuracy * 100))

    return test_accuracy


def main(args):

    if len(args.seeds) > 1:

        lst_test_accs = []

        names = ['Experiment_name', 'Network', 'Dataset', 'Mode', 'Seed', 'Accuracy']

        for seed in args.seeds:
            #model_dir = os.path.join(args.model_dir, args.model_dir + '_seed' + str(seed), 'checkpoints')
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

            test_loss, test_accuracy = solver(args, args.model_path)
            lst_test_accs.append(test_accuracy)

            values = [args.experiment_name, args.model_architecture, args.dataset, args.ft_prior if args.train_ftprior else 'Std', seed, test_accuracy]
    else:
        # model_dir = os.path.join(args.model_dir, os.path.basename(args.model_dir))
        # model_dir = os.path.join(model_dir + '_seed' + str(args.seeds[0]), 'checkpoints')
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

        utils.set_torch_seeds(args.seeds[0])
        test_accuracy = solver(args)

        names = ['Experiment_name', 'Network', 'Dataset', 'Train_Mode', 'Seed', 'Test_mode', 'Accuracy']
        values = [args.model_path, args.model_architecture, args.dataset, args.mode, args.seeds[0], args.ft_prior, test_accuracy]

    file = results_csv(args)
    if not args.append_csv:
        np.savetxt(file, (names, values), delimiter=',', fmt='%s')
        append_csv = True
    else:
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)



if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
