import argparse
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from kd_lib import losses as kd_losses
from train import adversarial_loss as adv_losses
from kd_lib import utilities as utils
from models.selector import select_model
from utilities import dist_utils
from utilities.results import save_results_normal
from utilities.utils import ModelSaver, check_final
from data.dataset import DATASETS
from data.transforms import build_transforms

parser = argparse.ArgumentParser(description='Baseline Training')
# Model options
parser.add_argument('--exp_identifier', type=str, default='')
parser.add_argument('--model_architecture', type=str, default='ResNet18')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--resnet_multiplier', default=1, type=int)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
# Dataset
parser.add_argument("--data_path", type=str, default='data')
parser.add_argument("--data_path_style", type=str, default=''),
parser.add_argument("--cache-dataset", action="store_true", default=False)
# Training options
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run for training the network')
parser.add_argument('--epoch_step', nargs='*', type=int, default=[60, 120, 160], help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--checkpoint', default='', type=str)
parser.add_argument('--seeds', nargs='*', type=int, default=[0, 10])
parser.add_argument('--mode', default='normal', choices=['normal', 'madry', 'trades'])
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--scheduler', default='None', choices=['None','cosine', 'multistep'])

# Device options
parser.add_argument('--cuda', action='store_true')
# evaluation options
parser.add_argument("--save_freq", default=10, type=int, help="save frequency")
parser.add_argument("--train_eval_freq",  default=10, type=int, help="evaluation frequency")
parser.add_argument("--test_eval_freq",  default=10, type=int, help="evaluation frequency")
# storage options
parser.add_argument('--dataroot', default='data', type=str)
parser.add_argument('--output_dir', default='experiments', type=str)
parser.add_argument('--checkpoint_dir', default='', type=str)
# Feature prior options
parser.add_argument("--train_ftprior", action="store_true", default=False)
parser.add_argument('--ft_prior', type=str, default='std', choices=['std', 'sobel'])
parser.add_argument('--norm_std', type=str, default='False')
parser.add_argument('--norm_fp', type=str, default='False')
# Sobel filter options
parser.add_argument('--sobel_gauss_ksize', default=3, type=int)
parser.add_argument('--sobel_ksize', default=3, type=int)
parser.add_argument("--sobel_upsample", type=str, default='False')
# Adversarial training
parser.add_argument('--step_size', type=float, default=0.007)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--perturb_steps', type=int,  default=10)
parser.add_argument('--distance', default='l_inf', choices=['l_2', 'l_inf'])
parser.add_argument('--trades_beta', type=float, default=5)
parser.add_argument('--mixup_alpha', type=float, default=1)
parser.add_argument('--sev', type=int,  default=1)
parser.add_argument('--cmnist_mode', default='fg', choices=['fg', 'bg', 'comb'])

# ======================================================================================
# Helper Functions
# ======================================================================================
def train_kd(args, model, device, train_loader, optimizer, epoch, writer):

    model.train()

    train_loss = 0
    correct = 0
    total = 0

    num_batches = len(train_loader)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc='batch training', total=num_batches):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out = model(data)
        iteration = (epoch * num_batches) + batch_idx

        if args.mode == 'normal':
            loss = kd_losses.cross_entropy(out, target)
            writer.add_scalar('train/loss', loss.item(), iteration)

        elif args.mode == 'madry':
            loss, _, _ = adv_losses.madry_loss(model, data, target, optimizer)

            writer.add_scalar('train/loss', loss.item(), iteration)

        elif args.mode == 'trades':

            loss, _, _ = adv_losses.trades_loss(model, data, target, optimizer, beta=args.trades_beta)
            writer.add_scalar('train/loss', loss.item(), iteration)

        else:
            raise ValueError('Incorrect Method selected')

        # perform back propagation
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(out.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().float().sum()
        b_idx = batch_idx

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))


# ======================================================================================
# Training Function
# ======================================================================================
def solver(args):

    log_dir = os.path.join(args.experiment_name, 'logs')
    model_dir = os.path.join(args.experiment_name, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    writer = SummaryWriter(log_path)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: %s' % device)

    if use_cuda:
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    if args.dataset =='imagenet' or args.dataset =='imagenet200':
        ds_class = DATASETS[args.dataset](args.data_path, cache_dataset=True)
    elif args.dataset =='style_tiny':
        ds_class = DATASETS[args.dataset](args.data_path, args.data_path_style)
    elif args.dataset =='col_mnist':
        ds_class = DATASETS[args.dataset](args.data_path, args.cmnist_mode)
    elif args.dataset =='cor_tinyimagenet':
        ds_class = DATASETS[args.dataset](args.data_path, args.sev)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)
    #load transforms
    transform_train, transform_test = build_transforms(args, ds_class)
    # load dataset
    trainset = ds_class.get_dataset('train', transform_train, transform_test)
    testset = ds_class.get_dataset('test', transform_train, transform_test)

    #data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    # Load model
    args.num_classes = ds_class.NUM_CLASSES
    cifar_resnet = True
    if args.dataset == 'imagenet200':
        cifar_resnet = False
    model = select_model(args.model_architecture, args.num_classes, cifar_resnet).to(device)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    saver = ModelSaver(model_dir)
    #csvwriter = csvWriter(args)

    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_step, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    start_epoch = 0
    if os.listdir(model_dir):
        model, optimizer, start_epoch = saver.load_checkpoint(model, optimizer, model_dir)

    print('*' * 60 + '\nTraining Mode: %s\n' % args.mode + '*' * 60)
    for epoch in tqdm(range(1, args.epochs + 1), desc='training epochs'):

        if epoch <= start_epoch:
            continue

        # adjust learning rate for SGD
        if scheduler:
            scheduler.step()
        else:
            utils.adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer)
        train_kd(args, model, device, train_loader, optimizer, epoch, writer)

        # evaluation on natural examples
        if epoch % args.train_eval_freq == 0:
            print("================================================================")
            train_loss, train_accuracy, correct = utils.eval(model, device, train_loader)
            print("Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                train_loss, correct, len(train_loader.dataset), train_accuracy * 100))
            print("================================================================")
            writer.add_scalar("Train/train_loss", train_loss, epoch)
            writer.add_scalar("Train/train_accuracy", train_accuracy, epoch)

        if epoch % args.test_eval_freq == 0:
            print("================================================================")
            test_loss, test_accuracy, correct = utils.eval(model, device, test_loader)
            print("Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                test_loss, correct, len(test_loader.dataset), test_accuracy * 100))
            print("================================================================")
            print("================================================================")
            writer.add_scalar("Test/test_loss", test_loss, epoch)
            writer.add_scalar("Test/test_accuracy", test_accuracy, epoch)

            if epoch != args.epochs:
                saver.save_models(model, optimizer, epoch, test_accuracy)


    # get final test accuracy
    test_loss, test_accuracy, correct = utils.eval(model, device, test_loader)
    writer.close()

    # save model
    torch.save(model, os.path.join(model_dir, 'final_model.pth'))

    return test_loss, test_accuracy, saver.best, saver.best_epoch


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    base_name = "%s_%s_%s_mode_%s_%sepochs" % (args.exp_identifier, args.model_architecture, args.mode, args.dataset, args.epochs)

    base_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # save training arguments
    args_path = os.path.join(base_dir, 'args.txt')
    z = vars(args).copy()
    with open(args_path, 'w') as f:
        f.write('arguments: ' + json.dumps(z) + '\n')

    if len(args.seeds) > 1:

        lst_test_accs = []
        lst_test_loss = []
        lst_best_accs = []
        lst_best_epochs = []
        for seed in args.seeds:
            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            utils.set_torch_seeds(seed)
            args.experiment_name = os.path.join(args.output_dir, base_name, base_name + '_seed' + str(seed))
            test_loss, test_accuracy, best_accuracy, best_epoch = solver(args)
            lst_test_loss.append(test_loss)
            lst_test_accs.append(test_accuracy)
            lst_best_accs.append(best_accuracy)
            lst_best_epochs.append(best_epoch)

        mu = np.mean(lst_test_accs)
        sigma = np.std(lst_test_accs)

        for i in range(len(args.seeds)):
            save_results_normal(args, os.path.join(args.output_dir, base_name, 'results.csv'),
                        lst_test_loss[i], lst_test_accs[i], lst_best_epochs[i], lst_best_accs[i], args.seeds[i], mu, sigma)

    else:
        utils.set_torch_seeds(args.seeds[0])
        args.experiment_name = os.path.join(args.output_dir, base_name, base_name + '_seed' + str(args.seeds[0]))

        if check_final(args):
            exit()

        test_loss, test_accuracy, best_accuracy, best_epoch = solver(args)
        save_results_normal(args, os.path.join(args.output_dir, base_name, 'results.csv'),
                                    test_loss, test_accuracy, best_epoch, best_accuracy)


if __name__ == '__main__':
    args = parser.parse_args()
    dist_utils.init_distributed_mode(args)

    main(args)
