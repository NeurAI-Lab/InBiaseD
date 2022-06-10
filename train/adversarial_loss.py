import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


# =============================================================================
# Adversarial Training
# =============================================================================
def trades_loss_v0(
        model,
        x_natural,
        y,
        optimizer,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        beta=1.0,
        distance='l_inf'
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
                loss_kl = criterion_kl(F.log_softmax(out_adv, dim=1),
                                       F.softmax(out_nat, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

                loss = (-1) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                           F.softmax(out_nat, dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    out_nat = model(x_natural)
    out_nat = out_nat[0] if isinstance(out_nat, tuple) else out_nat

    loss_natural = F.cross_entropy(out_nat, y)

    out_adv = model(x_adv)
    out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(out_adv, dim=1),
                                                    F.softmax(out_nat, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss, out_adv, x_adv


def madry_loss(model1,
               x1,
               y,
               optimizer,
               step_size=0.003,
               epsilon=0.031,
               perturb_steps=10,
               reduce=True
               ):

    model1.eval()

    # generate adversarial example
    x_adv = x1.detach() + 0.001 * torch.randn(x1.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()

        with torch.enable_grad():
            # loss_kl = F.cross_entropy(F.log_softmax(model(x_adv), dim=1), y)

            out_adv = model1(x_adv)
            out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
            loss = F.cross_entropy(out_adv, y)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x1 - epsilon), x1 + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model1.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    out_adv = model1(x_adv)
    out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv

    if reduce:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    loss_adv = criterion(out_adv, y)

    return loss_adv, out_adv, x_adv


# =============================================================================
# Robustness Evaluation
# =============================================================================
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device,
                  save_imgs = False,
                  ind = 0
                  ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    if save_imgs:
        visualize(X, X_pgd, random_noise, epsilon, ind)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_robustness(
    model,
    data_loader,
    epsilon,
    num_steps,
    step_size,
    random=True,
    device='cuda',
    save_imgs = False
):

    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    ind = 0

    if data_loader.dataset.transform_fp is None:
        for data, target in tqdm(data_loader, desc='robustness'):
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind)
            robust_err_total += err_robust
            natural_err_total += err_natural
            ind += 65
    else:
        for data, _, target in tqdm(data_loader, desc='robustness'):
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust, adv_imgs = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind)
            robust_err_total += err_robust
            natural_err_total += err_natural
            ind += 65
            #
            # if save_imgs:
            #     for i in range(len(adv_imgs)):
            #         from data.utils import plot
            #         orig = X[i].squeeze().detach().cpu().numpy()
            #         plot(orig, 'orig_%s' % i)
            #         adv = adv_imgs[i].squeeze().detach().cpu().numpy()
            #         plot(adv, 'adv_%s' % i)

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_samples = len(data_loader.dataset)

    rob_acc = (total_samples - successful_attacks) / total_samples
    nat_acc = (total_samples - nat_err) / total_samples

    print('=' * 30)
    print(f"Adversarial Robustness = {rob_acc * 100} % ({total_samples - successful_attacks}/{total_samples})")
    print(f"Natural Accuracy = {nat_acc * 100} % ({total_samples - nat_err}/{total_samples})")

    return nat_acc, rob_acc


def visualize(X, X_adv, X_grad, epsilon, ind):

    X = X.to('cpu')
    X_adv = X_adv.to('cpu')
    X_grad = X_grad.to('cpu')
    dst = "/volumes2/feature_prior_project/art/feature_prior/vis/adv2/base"

    for i in range(X.shape[0]):
        x = X[i].squeeze().detach().cpu().numpy()  # remove batch dimension # B X C H X W ==> C X H X W
        # x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        #     torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op- "unnormalize"
        #x = upsamp(x)
        x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x = np.clip(x, 0, 1)

        x_adv = X_adv[i].squeeze().detach().cpu().numpy()
        # x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
        #     torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
        #x_adv = upsamp(x_adv)
        x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)

        x_grad = X_grad[i].squeeze().detach().cpu().numpy()
        x_grad = np.transpose(x_grad, (1, 2, 0))
        x_grad = np.clip(x_grad, 0, 1)
        #x_grad = upsamp(x_grad)
        diff = (x - x_adv) * 30
        diff = np.clip(diff, 0, 1)
        imsave(os.path.join(dst, 'noise_%i.png' % (i+ind)), diff, format="png")
        imsave(os.path.join(dst, 'noisegrad_%i.png' % (i+ind)), x_grad, format="png")

        figure, ax = plt.subplots(1, 3, figsize=(7, 3))
        ax[0].imshow(x)
        ax[0].set_title('Clean Example', fontsize=10)
        imsave(os.path.join(dst, 'orig_%i.png' % (i+ind)), x, format="png")

        ax[1].imshow(diff)
        ax[1].set_title('Perturbation', fontsize=10)
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(x_adv)
        ax[2].set_title('Adversarial Example', fontsize=10)
        imsave(os.path.join(dst, 'adv_%i.png' % (i+ind)), x_adv, format="png")

        ax[0].axis('off')
        ax[2].axis('off')

        ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
                   transform=ax[0].transAxes)

        #plt.show()
        figure.savefig(os.path.join(dst, '%i.png' % (i+ind)), dpi=300, bbox_inches='tight')

from PIL import Image
import torchvision.transforms as transforms
def upsamp(img):

    curr_size = img.shape[0]
    upsample_size = 224
    resize_up = transforms.Resize(max(curr_size, upsample_size), 3)
    img = transforms.ToPILImage()(img)
    upsamp_img = np.array(resize_up(Image.fromarray(img)))
    return upsamp_img
