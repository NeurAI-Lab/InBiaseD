import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def eval(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)

    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy, correct

def test_inbiased(model1, model2, device, data_loader):
    model1.eval()
    model2.eval()

    loss_m1 = 0
    loss_m2 = 0
    correct_m1 = 0
    correct_m2 = 0
    with torch.no_grad():
        for data1, data2, target in tqdm(data_loader, desc='testing'):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output_m1 = model1(data1)
            output_m2 = model2(data2)

            #model1
            loss_m1 += F.cross_entropy(output_m1, target).item()
            pred_m1 = output_m1.max(1, keepdim=True)[1]
            correct_m1 += pred_m1.eq(target.view_as(pred_m1)).sum().item()
            #model2
            loss_m2 += F.cross_entropy(output_m2, target).item()
            pred_m2 = output_m2.max(1, keepdim=True)[1]
            correct_m2 += pred_m2.eq(target.view_as(pred_m2)).sum().item()

    loss_m1 /= len(data_loader.dataset)
    accuracy_m1 = correct_m1 / len(data_loader.dataset)

    loss_m2 /= len(data_loader.dataset)
    accuracy_m2 = correct_m2 / len(data_loader.dataset)

    return loss_m1, accuracy_m1, correct_m1, loss_m2, accuracy_m2, correct_m2


def adjust_learning_rate(epoch, epoch_steps, epoch_decay, optimizer):
    """decrease the learning rate"""
    
    if epoch in epoch_steps:
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = current_lr * epoch_decay
        print('=' * 60 + '\nChanging learning rate to %g\n' % (current_lr * epoch_decay) + '=' * 60)


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    # Note: Input model & optimizer should be pre-defined.  This routine only
    # updates their states.

    start_epoch = 0

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


# Rampup Methods
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length