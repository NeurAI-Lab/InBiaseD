import argparse
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from utilities.results import save_results_inbiased
from torch.utils.tensorboard import SummaryWriter
from kd_lib import losses as kd_losses
from kd_lib import utilities as utils
from models.selector import select_model
from data.dataset import DATASETS
from data.transforms import build_transforms
from utilities.utils import ModelSaver, check_final
from kd_lib.losses import inbiased_Loss, collate_loss

parser = argparse.ArgumentParser(description='InBiaseD Training')
# Model options
parser.add_argument('--exp_identifier', type=str, default='')
parser.add_argument('--model1_architecture', type=str, default='ResNet18')
parser.add_argument('--model2_architecture', type=str, default='ResNet18')
parser.add_argument('--resnet_multiplier', type=int, default=1)
# Dataset
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--use_imbalanced_dataset', action='store_true', default=False)
parser.add_argument('--imbalance_gamma', type=float, default=1)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'stl10', 'stltint', 'tinyimagenet', 'imagenet', 'imagenet200', 'celeba', 'delaunay', 'col_mnist', 'cor_cifar10', 'cor_tinyimagenet'])
parser.add_argument('--label_corrupt_prob', default=0, type=float)
parser.add_argument("--data_path", type=str, default='data'), \
parser.add_argument("--cache-dataset", action="store_true", default=False)
# Training options
parser.add_argument('--dtype', type=str, default='float')
parser.add_argument('--nthread', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run for training the network')
parser.add_argument('--epoch_step', nargs='*', type=int, default=[60, 120, 160], help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--seeds', nargs='*', type=int, default=[0, 10])
parser.add_argument('--scheduler', default='None', choices=['None', 'cosine', 'multistep'])

# knowledge distillation methods
parser.add_argument('--mode', default='inbiased', choices=['inbiased',])
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--loss_type', nargs='*', type=str, default=['kl'], help="--loss_type kl at")
parser.add_argument('--loss_dynamic', action='store_true')
parser.add_argument('--loss_schedule', nargs='*', type=int, default=[120, 160])
parser.add_argument('--loss_decay', nargs='*', type=float, default=[0.1, 0.01])
parser.add_argument('--loss_wt_kl1', type=float, default='1.0')
parser.add_argument('--loss_wt_at1', type=float, default='1.0')
parser.add_argument('--loss_wt_l21', type=float, default='1.0')
parser.add_argument('--loss_wt_kl2', type=float, default='1.0')
parser.add_argument('--loss_wt_at2', type=float, default='1.0')
parser.add_argument('--loss_wt_l22', type=float, default='1.0')
# Feature prior options
parser.add_argument("--train_ftprior", action="store_true", default=False)
parser.add_argument('--ft_prior', type=str, default='std', choices=['std', 'sobel'])
parser.add_argument('--norm_std', type=str, default='False')
parser.add_argument('--norm_fp', type=str, default='False')
# Device options
parser.add_argument('--cuda', action='store_true')
# evaluation options
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--train_eval_freq", type=int,  default=10)
parser.add_argument("--test_eval_freq",  type=int, default=10)
# storage options
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--output_dir',  type=str, default='experiments')
parser.add_argument('--model1_dir', default='', type=str)
parser.add_argument('--model2_dir', default='', type=str)
#Adv
parser.add_argument("--train_adv", action="store_true", default=False)
# Sobel filter options
parser.add_argument('--sobel_gauss_ksize', default=3, type=int)
parser.add_argument('--sobel_ksize', default=3, type=int)
parser.add_argument("--sobel_upsample", type=str, default='True')
parser.add_argument('--sev', type=int,  default=1)
parser.add_argument('--cmnist_mode', default='fg', choices=['fg', 'bg', 'comb'])

# ======================================================================================
# Helper Functions
# ======================================================================================

def train_inbiased(args, model1, model2, device, train_loader, optimizer_m1, optimizer_m2, epoch, writer):

    model1.train()
    model2.train()

    m1_train_loss = 0
    m1_correct = 0

    m2_train_loss = 0
    m2_correct = 0

    total = 0

    num_batches = len(train_loader)

    for batch_idx, (data1, data2, target) in tqdm(enumerate(train_loader), desc='batch training', total=num_batches):

        data1, data2, target = data1.to(device), data2.to(device), target.to(device)

        # reset the gradients
        optimizer_m1.zero_grad()
        optimizer_m2.zero_grad()

        iteration = (epoch * num_batches) + batch_idx

        m1_out = model1(data1)
        m2_out = model2(data2)

        l_ce_m1 = kd_losses.cross_entropy(m1_out, target)
        l_ce_m2 = kd_losses.cross_entropy(m2_out, target)

        inbiased_loss_m1 = inbiased_Loss(args, model1.feat, [ft.detach() for ft in model2.feat],
                               m1_out, m2_out.detach())
        inbiased_loss_m2 = inbiased_Loss(args, model2.feat, [ft.detach() for ft in model1.feat],
                               m2_out, m1_out.detach())

        loss_m1_dict = collate_loss(args, loss_ce=l_ce_m1, loss_inbiased=inbiased_loss_m1.loss_dml, m1=True, epoch=epoch)
        loss_m2_dict = collate_loss(args, loss_ce=l_ce_m2, loss_inbiased=inbiased_loss_m2.loss_dml, m1=False, epoch=epoch)

        loss_m1 = loss_m1_dict['loss']
        loss_m2 = loss_m2_dict['loss']

        loss_m1.backward()
        optimizer_m1.step()

        loss_m2.backward()
        optimizer_m2.step()

        for loss_name, loss_item in loss_m1_dict.items():
            writer.add_scalar('Model1 losses/{}'.format(loss_name), loss_item,
                                      global_step=iteration)
        for loss_name, loss_item in loss_m2_dict.items():
            writer.add_scalar('Model2 losses/{}'.format(loss_name), loss_item,
                                      global_step=iteration)

        m1_train_loss += loss_m1.data.item()
        m2_train_loss += loss_m2.data.item()

        _, predicted_m1 = torch.max(m1_out.data, 1)
        _, predicted_m2 = torch.max(m2_out.data, 1)

        m1_correct += predicted_m1.eq(target.data).cpu().float().sum()
        m2_correct += predicted_m2.eq(target.data).cpu().float().sum()
        total += target.size(0)

        m1_train_loss /= num_batches + 1
        m1_acc = 100.0 * m1_correct / total

        m2_train_loss /= num_batches + 1
        m2_acc = 100.0 * m2_correct / total

    print("Model 1 Loss: %.3f | Acc: %.3f%% (%d/%d)" % (m1_train_loss, m1_acc, m1_correct, total))
    print("Model 2 Loss: %.3f | Acc: %.3f%% (%d/%d)" % (m2_train_loss, m2_acc, m2_correct, total))


# ======================================================================================
# Training Function
# ======================================================================================
def solver(args):

    print(args.experiment_name)

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

    # load dataset
    if args.dataset =='imagenet' or args.dataset =='imagenet200':
        ds_class = DATASETS[args.dataset](args.data_path, cache_dataset=True)
    elif args.dataset =='cor_tinyimagenet':
        ds_class = DATASETS[args.dataset](args.data_path, args.sev)
    elif args.dataset =='col_mnist':
        ds_class = DATASETS[args.dataset](args.data_path, args.cmnist_mode)
    else:
        ds_class = DATASETS[args.dataset](args.data_path)
    ft_prior = args.ft_prior
    args.ft_prior = 'std'
    transform_train, transform_test = build_transforms(args, ds_class)
    args.ft_prior = ft_prior
    transform_train_fp, transform_test_fp = build_transforms(args, ds_class)
    trainset = ds_class.get_dataset('train', transform_train, transform_test, transform_train_fp, transform_test_fp)
    testset = ds_class.get_dataset('test', transform_train, transform_test, transform_train_fp, transform_test_fp)

    print("==> Preparing data..")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print("Dataset Loaded")

    # Initialize Models
    args.num_classes = ds_class.NUM_CLASSES

    cifar_resnet = True
    if args.dataset == 'imagenet200':
        cifar_resnet = False
    model1 = select_model(args.model1_architecture, args.num_classes, cifar_resnet).to(device)
    model2 = select_model(args.model2_architecture, args.num_classes, cifar_resnet).to(device)

    optimizer_m1 = SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer_m2 = SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler_m1 = None
    scheduler_m2 = None
    if args.scheduler == 'multistep':
        scheduler_m1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_m1, milestones=args.epoch_step, gamma=0.1)
        scheduler_m2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_m2, milestones=args.epoch_step, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler_m1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m1, T_max=args.epochs)
        scheduler_m2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m2, T_max=args.epochs)

    saver_m1 = ModelSaver(model_dir, base_name='model1')
    saver_m2 = ModelSaver(model_dir, base_name='model2')
    start_epoch = 0
    if os.listdir(model_dir):
        model1, optimizer_m1, start_epoch = saver_m1.load_checkpoint(model1, optimizer_m1, model_dir)
        model2, optimizer_m2, start_epoch = saver_m2.load_checkpoint(model2, optimizer_m2, model_dir)

    print('*' * 60 + '\nTraining the model with %s\n' % args.mode + '*' * 60)
    for epoch in tqdm(range(1, args.epochs + 1), desc='training epochs'):

        if epoch <= start_epoch:
            continue

        # adjust learning rate for SGD
        # adjust learning rate for SGD
        if scheduler_m1:
            scheduler_m1.step()
            scheduler_m2.step()
        else:
            utils.adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer_m1)
            utils.adjust_learning_rate(epoch, args.epoch_step, args.lr_decay_ratio, optimizer_m2)

        train_inbiased(args, model1, model2, device, train_loader, optimizer_m1, optimizer_m2, epoch, writer)

        # evaluation on natural examples
        if epoch % args.train_eval_freq == 0:
            print("================================================================")
            train_loss_m1, train_accuracy_m1, correct_m1,\
                train_loss_m2, train_accuracy_m2, correct_m2 = utils.test_inbiased(model1, model2, device, train_loader)

            print("Model 1 Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                train_loss_m1, correct_m1, len(train_loader.dataset), train_accuracy_m1 * 100))
            print("================================================================")
            writer.add_scalar("Train/m1_train_loss", train_loss_m1, epoch)
            writer.add_scalar("Train/m1_train_accuracy", train_accuracy_m1, epoch)

            print("Model 2 Training: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                train_loss_m2, correct_m2, len(train_loader.dataset), train_accuracy_m2 * 100))
            print("================================================================")
            writer.add_scalar("Train/m2_train_loss", train_loss_m2, epoch)
            writer.add_scalar("Train/m2_train_accuracy", train_accuracy_m2, epoch)

        if epoch % args.test_eval_freq == 0:
            print("================================================================")
            test_loss_m1, test_accuracy_m1, correct_m1, \
                test_loss_m2, test_accuracy_m2, correct_m2 = utils.test_inbiased(model1, model2, device, test_loader)

            print("Model 1 Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                test_loss_m1, correct_m1, len(test_loader.dataset), test_accuracy_m1 * 100))
            print("================================================================")
            writer.add_scalar("Test/m1_test_loss", test_loss_m1, epoch)
            writer.add_scalar("Test/m1_test_accuracy", test_accuracy_m1, epoch)
            if epoch != args.epochs:
                saver_m1.save_models(model1, optimizer_m1, epoch, test_accuracy_m1)

            print("Model 2 Test: Average loss: {:.4f}, Accuracy: {}/{} ({}%)".format(
                test_loss_m2, correct_m2, len(test_loader.dataset)
                , test_accuracy_m2 * 100))
            print("================================================================")
            writer.add_scalar("Test/m2_test_loss", test_loss_m2, epoch)
            writer.add_scalar("Test/m2_test_accuracy", test_accuracy_m2, epoch)
            if epoch != args.epochs:
                saver_m2.save_models(model2, optimizer_m2, epoch, test_accuracy_m2)

    # get final test accuracy
    #test_loss, test_accuracy, correct = utils.eval(model1, device, test_loader)
    test_loss_m1, test_accuracy_m1, correct_m1, \
    test_loss_m2, test_accuracy_m2, correct_m2 = utils.test_inbiased(model1, model2, device, test_loader)
    writer.close()

    # save model
    torch.save(model1, os.path.join(model_dir, 'final_model1.pth'))
    torch.save(model2, os.path.join(model_dir, 'final_model2.pth'))

    return test_loss_m1, test_accuracy_m1, saver_m1.best, saver_m1.best_epoch, test_loss_m2, test_accuracy_m2


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    base_name = "%s_%s_%s_%s_%sep" % (args.exp_identifier, args.mode, args.model1_architecture, args.dataset, args.epochs)

    base_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # save training arguments
    args_path = os.path.join(base_dir, 'args.txt')
    z = vars(args).copy()
    with open(args_path, 'w') as f:
        f.write('arguments: ' + json.dumps(z) + '\n')

    if len(args.seeds) > 1:

        lst_test_accs_m1 = []
        lst_test_accs_m2 = []
        lst_test_loss_m1 = []
        lst_test_loss_m2 = []
        lst_best_accs_m1 = []
        lst_best_epochs_m1 = []
        for seed in args.seeds:

            print('\n\n----------- SEED {} -----------\n\n'.format(seed))
            utils.set_torch_seeds(seed)
            args.experiment_name = os.path.join(args.output_dir, base_name, base_name + '_seed' + str(seed))
            test_loss_m1, test_accuracy_m1, best_accuracy_m1, best_epoch_m1, \
            test_loss_m2, test_accuracy_m2 = solver(args)

            lst_test_loss_m1.append(test_loss_m1)
            lst_test_accs_m1.append(test_accuracy_m1)
            lst_test_loss_m2.append(test_loss_m2)
            lst_test_accs_m2.append(test_accuracy_m2)
            lst_best_accs_m1.append(best_accuracy_m1)
            lst_best_epochs_m1.append(best_epoch_m1)

        mu = np.mean(lst_test_accs_m1)
        sigma = np.std(lst_test_accs_m1)

        for i in range(len(args.seeds)):
            save_results_inbiased(args, os.path.join(args.output_dir, base_name, 'results.csv'), lst_test_loss_m1[i],
                             lst_test_accs_m1[i], lst_best_epochs_m1[i], lst_best_accs_m1[i], lst_test_loss_m2[i], lst_test_accs_m2[i], args.seeds[i], mu, sigma)
    else:
        utils.set_torch_seeds(args.seeds[0])
        args.experiment_name = os.path.join(args.output_dir, base_name, base_name + '_seed' + str(args.seeds[0]))
        if check_final(args):
            exit()

        test_loss_m1, test_accuracy_m1, best_accuracy_m1, best_epoch_m1, \
        test_loss_m2, test_accuracy_m2 = solver(args)

        save_results_inbiased(args, os.path.join(args.output_dir, base_name, 'results.csv'), test_loss_m1, test_accuracy_m1,
                         best_epoch_m1, best_accuracy_m1, test_loss_m2, test_accuracy_m2)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
