import os
import csv
import random
import os.path
import shutil
import torch
import numpy as np
from utilities import dist_utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_data_loader(dataset, batch_size, cuda=False, shuffle=True):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch, precision, best=True, save_splits=False):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    checkpoint_dict = {
        'state': model.state_dict(),
        'epoch': epoch,
        'precision': precision,

    }

    if save_splits:
        checkpoint_dict['split_level'] = model.split_level
        checkpoint_dict['in_perm'] = model.in_perm
        checkpoint_dict['out_perm'] = model.out_perm

    torch.save(checkpoint_dict, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision


def load_checkpoint_from_path(model, checkpoint_path):
    # load the checkpoint.
    checkpoint = torch.load(checkpoint_path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(checkpoint_path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    precision = checkpoint['precision']
    return epoch, precision

class ModelSaver:

    def __init__(self, save_root, base_name='model',
                 best_name='best', latest_name='latest',
                 models_to_keep=1):
        self.save_root = save_root
        self.base_name = base_name
        self.best_name = best_name
        self.latest_name = latest_name
        self.models_to_keep = models_to_keep
        self.best = 0
        self.best_epoch = 0

    def read_models(self):
        task_to_models = []

        for file in os.listdir(self.save_root):
            if os.path.isdir(file):
                continue
            extension = file.split('.')[-1]
            if extension != 'pth':
                continue
            names = file.split('_')[1:-1]
            if names[0] == 'best':
                continue
            task = '_'.join(names)
            task_to_models.append(file)

        return task_to_models

    def delete_old(self, task_to_models):
        models = sorted(task_to_models)
        if len(models) > self.models_to_keep:
            for m in models[:-self.models_to_keep]:
                os.remove(os.path.join(self.save_root, m))

    def save(self, save_dict, epoch, name, accuracy):
        accuracy *= 100

        if name == 'best':
            for file in os.listdir(self.save_root):
                if os.path.isdir(file):
                    continue
                extension = file.split('.')[-1]
                if extension != 'pth':
                    continue
                names = file.split('_')[:]
                if names[0] == self.base_name and names[1] == 'best':
                    os.remove(os.path.join(self.save_root, file))

        file_name = f'{self.base_name}_{name}_epoch{epoch:03d}_acc{accuracy:02.2f}.pth'
        dist_utils.save_on_master(save_dict, os.path.join(self.save_root, file_name),
                                  _use_new_zipfile_serialization=False)

    def save_models(self, model, optimizer, epoch, accuracy):

        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'precision': accuracy,

        }

        if accuracy > self.best:
            self.best = accuracy
            self.best_epoch = epoch
            self.save(save_dict, epoch, self.best_name, accuracy)
        else:
            self.save(save_dict, epoch, self.latest_name, accuracy)

        task_to_models = self.read_models()
        self.delete_old(task_to_models)

    def load_checkpoint(self, model, optimizer, model_dir, best=False):
        model_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(model_dir)))
        best_files = []
        for f in model_files:
            if 'best' in f:
                best_files.append(f)
        files = []
        for f in best_files:
            if self.base_name in f:
                files.append(f)

        # best_files = [f if 'best' in f else None for f in model_files]
        # file = [f if self.base_name in f else None for f in best_files]

        for f in files:
            if f is not None:
                filename = os.path.join(model_dir, f)

        # load the checkpoint.
        if not os.path.isfile(filename):
            print("=> no checkpoint found at '{}'".format(filename))

        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))

        return model, optimizer, start_epoch

def validate(args, model, dataset, cuda=False, verbose=True):
    # set model mode as test mode.
    model_mode = model.training
    model.train(mode=False)

    # prepare the data loader and the statistics.
    # data_loader = get_data_loader(dataset, 32, cuda=cuda)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False)
    total_tested = 0
    total_correct = 0

    for data, labels in data_loader:
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = torch.max(scores, 1)
        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model.train(mode=model_mode)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision

def validate_ensemble(model1, model2, dataset1, dataset2, cuda=False, verbose=True):
    # set model mode as test mode.
    model_mode1 = model1.training
    model1.train(mode=False)
    model_mode2 = model2.training
    model2.train(mode=False)

    # prepare the data loader and the statistics.
    data_loader1 = get_data_loader(dataset1, 32, cuda=cuda, shuffle=False)
    data_loader2 = get_data_loader(dataset2, 32, cuda=cuda, shuffle=False)
    total_tested = 0
    total_correct = 0

    for (data, labels), (data2, labels2) in zip(data_loader1, data_loader2):
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        data2 = Variable(data2).cuda() if cuda else Variable(data2)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores1 = model1(data)
        scores2 = model2(data2)

        # scores = (scores1+scores2)/2
        # scores = torch.max(scores1, scores2)
        scores1 = scores1[0] if isinstance(scores1, tuple) else scores1
        scores2 = scores2[0] if isinstance(scores2, tuple) else scores2

        softmaxes1 = F.softmax(scores1, dim=1)
        softmaxes2 = F.softmax(scores2, dim=1)
        scores = (softmaxes1+softmaxes2)/2
        # scores = torch.max(softmaxes1, softmaxes2)
        _, predicted = torch.max(scores, 1)

        # # scores = (scores1+scores2)/2
        # _, predicted = torch.max(scores, 1)

        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model1.train(mode=model_mode1)
    model2.train(mode=model_mode2)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision

def validate_ensemble_celeb(model1, model2, dataset1, dataset2, cuda=False, verbose=True):
    # set model mode as test mode.
    model_mode1 = model1.training
    model1.train(mode=False)
    model_mode2 = model2.training
    model2.train(mode=False)

    # prepare the data loader and the statistics.
    data_loader1 = get_data_loader(dataset1, 32, cuda=cuda, shuffle=False)
    data_loader2 = get_data_loader(dataset2, 32, cuda=cuda, shuffle=False)
    y = []
    y_pred = []
    is_blond = []

    for data, labels, blond in data_loader1:
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores1 = model1(data)
        scores2 = model2(data)

        # scores = (scores1+scores2)/2
        # scores = torch.max(scores1, scores2)
        scores1 = scores1[0] if isinstance(scores1, tuple) else scores1
        scores2 = scores2[0] if isinstance(scores2, tuple) else scores2

        softmaxes1 = F.softmax(scores1, dim=1)
        softmaxes2 = F.softmax(scores2, dim=1)
        scores = (softmaxes1+softmaxes2)/2
        _, predicted = torch.max(scores, 1)

        y.append(labels.cpu())
        y_pred.append(predicted.detach().cpu())
        is_blond.append(blond.cpu())

    y = torch.cat(y).numpy()
    y_pred = torch.cat(y_pred).numpy()
    is_blond = torch.cat(is_blond).numpy()

    np.mean(y == y_pred)

    # Men Blonde
    num_samples = len(y)
    blonde_male = [(y[i] == 1) and is_blond[i] for i in range(num_samples)]
    blonde_female = [(y[i] == 0) and is_blond[i] for i in range(num_samples)]
    non_blonde_male = [(y[i] == 1) and not is_blond[i] for i in range(num_samples)]
    non_blonde_female = [(y[i] == 0) and not is_blond[i] for i in range(num_samples)]

    print('Overall:',  np.mean(y == y_pred))
    print('Blonde Male:',  np.mean(y[blonde_male] == y_pred[blonde_male]))
    print('Non Blonde Male:',  np.mean(y[non_blonde_male] == y_pred[non_blonde_male]))
    print('Blonde Female:',  np.mean(y[blonde_female] == y_pred[blonde_female]))
    print('Non Blonde Female:',  np.mean(y[non_blonde_female] == y_pred[non_blonde_female]))

    overall = np.mean(y == y_pred)
    blond_male = np.mean(y[blonde_male] == y_pred[blonde_male])
    nonblonde_male = np.mean(y[non_blonde_male] == y_pred[non_blonde_male])
    blond_female = np.mean(y[blonde_female] == y_pred[blonde_female])
    nonblond_female = np.mean(y[non_blonde_female] == y_pred[non_blonde_female])

    return overall, blond_male, nonblonde_male, blond_female, nonblond_female

class csvWriter():
    def __init__(self, args):
        self.names = [
        "exp_identifier",
        "model",
        "mode",
        "dataset",
        "epochs",
        "accuracy",
        "output_path"
        ]

        base_name = "%s_%s_%s_mode_%s_%sepochs" % (
        args.exp_identifier, args.model_architecture, args.mode, args.dataset, args.epochs)

        self.file = os.path.join(args.output_dir, base_name, "results.csv")

        np.savetxt(self.file, (self.names), delimiter=",", fmt="%s")

    def write(self, args, test_accuracy):
        values = [
            args.exp_identifier,
            args.model_architecture,
            args.mode,
            args.dataset,
            args.epochs,
            test_accuracy,
            os.path.dirname(self.file)
        ]
        with open(self.file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(values)

def check_final(args):
    model_dir = os.path.join(args.experiment_name, 'checkpoints')
    if not os.path.exists(model_dir):
        return False
    else:
        model_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(model_dir)))

        for file in model_files:
            if 'final_model' in file:
                print("Model fully trained")
                return True
        else:
            return False

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
