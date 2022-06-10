import argparse
import os
import csv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utilities.utils import validate_ensemble, validate_ensemble_celeb
from kd_lib import utilities as utils
from models.selector import select_model
from data.transforms import build_transforms
from data.dataset import DATASETS

parser = argparse.ArgumentParser(description='Evaluation Script for classification')
# Model options
parser.add_argument('--exp_identifier', type=str, default='')
parser.add_argument('--mode', default='std', choices=['std', 'inbias'])
parser.add_argument('--model_architecture', type=str, default='ResNet18')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--resnet_multiplier', default=1, type=int)
parser.add_argument('--seeds', nargs='*', type=int, default=[0, 10])
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run for training the network')
# Dataset
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--cache-dataset", action="store_true", default=False)
# Device options
parser.add_argument('--cuda', action='store_true')
# evaluation options
parser.add_argument("--model_path1",  default='', type=str)
parser.add_argument("--model_path2",  default='', type=str)
parser.add_argument('--batch_size', default=1, type=int)
# storage options
parser.add_argument('--output_dir', default='experiments', type=str)
parser.add_argument("--results_csv_dir", default="None", type=str, help="specify a directory for results csv")
parser.add_argument("--append_csv", type=str, default='False', help="If append to existing csv")
# Feature prior options
parser.add_argument('--ft_prior', type=str, default='std', choices=['std', 'sobel'])
# Sobel filter options
parser.add_argument('--sobel_gauss_ksize', type=int, default=3 )
parser.add_argument('--sobel_ksize', type=int, default=3)
parser.add_argument("--sobel_upsample", type=str, default='True')
parser.add_argument('--norm_std', type=str, default='True')
parser.add_argument('--norm_fp', type=str, default='False')
parser.add_argument('--cmnist_mode', default='fg', choices=['fg', 'bg', 'comb'])

# ======================================================================================
# Helper Function
# ======================================================================================
def results_csv(args):

    if args.mode == 'std':
        name = 'ensemble_std.csv'
    else:
        name = 'ensemble_inbias.csv'

    if not args.append_csv:
        file = os.path.join(args.output_dir, name)
        if os.path.exists(file):
            os.remove(file)
    else:
        file = os.path.join(args.output_dir, name)
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
    if args.dataset =='col_mnist':
        ds_class = DATASETS[args.dataset](args.data_path, args.cmnist_mode)
    elif args.dataset =='imagenet' or args.dataset =='imagenet200':
        ds_class = DATASETS[args.dataset](args.data_path, cache_dataset=True)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)

    _, transform_test = build_transforms(args, ds_class)
    testset1 = ds_class.get_dataset('test', None, transform_test)

    if args.dataset =='col_mnist':
        ds_class = DATASETS[args.dataset](args.data_path, args.cmnist_mode)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)

    if args.mode == 'inbias':
        args.ft_prior = 'sobel' #'std' #
        args.norm_std = 'False'
    _, transform_test = build_transforms(args, ds_class)
    testset2 = ds_class.get_dataset('test', None, transform_test)

    # Load model
    model1 = select_model(args.model_architecture, args.num_classes).to(device)
    model2 = select_model(args.model_architecture, args.num_classes).to(device)

    checkpoint = torch.load(args.model_path1)
    if isinstance(checkpoint, dict):
        model1.load_state_dict(checkpoint["state"])
    else:
        model1 = checkpoint

    checkpoint = torch.load(args.model_path2)
    if isinstance(checkpoint, dict):
        model2.load_state_dict(checkpoint["state"])
    else:
        model2 = checkpoint

    # get final test accuracy
    test_accuracy = validate_ensemble(model1, model2, testset1, testset2, True)
    return test_accuracy

    #celeb
    # overall, blond_male, nonblonde_male, blond_female, nonblond_female = validate_ensemble_celeb(model1, model2, testset1, testset2, True)
    # print("Test: overall: ({}%)".format(overall))
    # print("Test: blond_male: ({}%)".format(blond_male ))
    # print("Test: nonblonde_male: ({}%)".format(nonblonde_male ))
    # print("Test: blond_female: ({}%)".format(blond_female ))
    # print("Test: nonblond_female: ({}%)".format(nonblond_female ))
    # return 0

def main(args):
    # model_dir = os.path.join(args.model_dir, os.path.basename(args.model_dir))
    # model_dir = os.path.join(model_dir + '_seed' + str(args.seeds[0]), 'checkpoints')
    os.makedirs(os.path.dirname(args.model_path1), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path2), exist_ok=True)

    utils.set_torch_seeds(args.seeds[0])
    #for fourier analysis
    # radii = [1,3]
    # for radius in radii:
    #     args.radius = radius
    test_accuracy = solver(args)
    names = ['Model1', 'Model2', 'Network', 'Dataset', 'Train_Mode', 'Seed', 'Test_mode', 'Accuracy']
    values = [args.model_path1, args.model_path2, args.model_architecture, args.dataset, args.mode, args.seeds[0], args.ft_prior, test_accuracy]

    file = results_csv(args)
    if not args.append_csv:
        np.savetxt(file, (names, values), delimiter=',', fmt='%s')
        args.append_csv = True
    else:
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)



if __name__ == '__main__':
    args = parser.parse_args()

    main(args)
