import torch
import random
import torch.nn.functional as F
from torch import nn

criterion_MSE = nn.MSELoss(reduction='mean')


def cross_entropy(y, labels):
    l_ce = F.cross_entropy(y, labels)
    return l_ce

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def get_loss_wt(args, epoch, loss_wt):

    if args.loss_dynamic:
        if epoch in args.loss_schedule:
            loss_wt *= args.loss_decay[args.loss_schedule.index(epoch)]
            return loss_wt

    return loss_wt

def collate_loss(args, loss_ce, loss_inbiased, loss_adv=0, m1=True, epoch=0):

    if m1:
        str = "m1"
    else:
        str = "m2"
    final_loss = {}
    final_loss[str +'_loss_ce'] = loss_ce
    final_loss['loss'] = 0

    if loss_inbiased is not None:
        if 'kl' in args.loss_type:
            loss_wt_kl = args.loss_wt_kl1 if m1 else args.loss_wt_kl2
            loss_wt_kl = get_loss_wt(args, epoch, loss_wt_kl)
            final_loss[str +'_loss_kl'] = loss_inbiased['loss_kl'] * loss_wt_kl
            loss_inbiased['loss'] += final_loss[str +'_loss_kl']

        if 'l2' in args.loss_type:
            loss_wt_l2 = args.loss_wt_l21 if m1 else args.loss_wt_l22
            loss_wt_l2 = get_loss_wt(args, epoch, loss_wt_l2)
            final_loss[str +'_loss_l2'] = loss_inbiased['loss_l2'] * loss_wt_l2
            loss_inbiased['loss'] += final_loss[str +'_loss_l2']

        if 'at' in args.loss_type:
            loss_wt_at = args.loss_wt_at1 if m1 else args.loss_wt_at2
            final_loss[str +'_loss_at'] = loss_inbiased['loss_at'] * loss_wt_at
            loss_inbiased['loss'] += final_loss[str +'_loss_at']

        if 'fitnet' in args.loss_type:
            loss_wt_at = args.loss_wt_at1 if m1 else args.loss_wt_at2
            final_loss[str +'_loss_fitnet'] = loss_inbiased['loss_fitnet'] * loss_wt_at
            loss_inbiased['loss'] += final_loss[str +'_loss_fitnet']

        final_loss['loss'] += loss_inbiased['loss']

    if args.train_adv:
        if 'madry' in args.adv_loss_type:
            final_loss[str + '_loss_adv'] = loss_adv
        if 'trades' in args.adv_loss_type:
            final_loss[str + '_loss_adv'] = loss_adv
        final_loss['loss'] += loss_adv

    final_loss['loss'] += loss_ce

    return final_loss


class inbiased_Loss():
    def __init__(self, args, st_feat,tch_feat, st_out, tch_out):
        self.args = args
        self.st_feat = st_feat
        self.tch_feat = tch_feat
        self.st_out = st_out
        self.tch_out = tch_out
        self.loss_dml = {}
        self.loss_dml['loss'] = 0
        self.get_loss()

    def get_loss(self):

        if 'kl' in self.args.loss_type:
            self.loss_dml['loss_kl'] = 0
            loss = self.kl_loss(self.st_out, self.tch_out, self.args.temperature)
            self.loss_dml['loss_kl'] = loss

        if 'at' in self.args.loss_type:
            self.loss_dml['loss_at'] = 0

            n = random.randint(3, 4)
            loss = self.at_loss(self.st_feat[n], self.tch_feat[n])
            self.loss_dml['loss_at'] = loss

        if 'fitnet' in self.args.loss_type:
            self.loss_dml['loss_fitnet'] = 0

            if self.args.model1_architecture == 'MLP':
                n = random.randint(0, 1)
                loss = self.fitnet_loss(self.st_feat[n], self.tch_feat[n])
            else:
                n = random.randint(3, 4)
                loss = self.fitnet_loss(self.st_feat[n], self.tch_feat[n])
            self.loss_dml['loss_fitnet'] = loss

        if 'l2' in self.args.loss_type:
            self.loss_dml['loss_l2'] = 0
            loss = self.l2_loss(self.st_out, self.tch_out)
            self.loss_dml['loss_l2'] = loss

    def kl_loss(self,student_scores, teacher_scores, T):

        p = F.log_softmax(student_scores / T, dim=1)
        q = F.softmax(teacher_scores / T, dim=1)

        l_kl = F.kl_div(p, q, size_average=False) * (T**2) / student_scores.shape[0]

        return l_kl

    def fitnet_loss(self, A_s, A_t, rand=False, noise=0.1):

        if rand:
            rand_noise =  torch.FloatTensor(A_t.shape).uniform_(1 - noise, 1 + noise)
            A_t = A_t * rand_noise

        return criterion_MSE(A_s, A_t)

    def l2_loss(self, ft_s, ft_t):
        return criterion_MSE(ft_s, ft_t)

    def at_loss(self, x, y, rand=False, noise=0.1):
        if rand:
            rand_noise = torch.FloatTensor(y.shape).uniform_(1 - noise, 1 + noise).cuda()
            y = y * rand_noise

        return (at(x) - at(y)).pow(2).mean()

    def FSP_loss(fea_t, short_t, fea_s, short_s, rand=False, noise=0.1):

        a, b, c, d = fea_t.size()
        feat = fea_t.view(a, b, c * d)
        a, b, c, d = short_t.size()
        shortt = short_t.view(a, b, c * d)
        G_t = torch.bmm(feat, shortt.permute(0, 2, 1)).div(c * d).detach()

        a, b, c, d = fea_s.size()
        feas = fea_s.view(a, b, c * d)
        a, b, c, d = short_s.size()
        shorts = short_s.view(a, b, c * d)
        G_s = torch.bmm(feas, shorts.permute(0, 2, 1)).div(c * d)

        return criterion_MSE(G_s, G_t)

    def similarity_preserving_loss(A_t, A_s):
        """Given the activations for a batch of input from the teacher and student
        network, calculate the similarity preserving knowledge distillation loss from the
        paper Similarity-Preserving Knowledge Distillation (https://arxiv.org/abs/1907.09682)
        equation 4

        Note: A_t and A_s must have the same batch size

        Parameters:
            A_t (4D tensor): activation maps from the teacher network of shape b x c1 x h1 x w1
            A_s (4D tensor): activation maps from the student network of shape b x c2 x h2 x w2

        Returns:
            l_sp (1D tensor): similarity preserving loss value
    """

        # reshape the activations
        b1, c1, h1, w1 = A_t.shape
        b2, c2, h2, w2 = A_s.shape
        assert b1 == b2, 'Dim0 (batch size) of the activation maps must be compatible'

        Q_t = A_t.reshape([b1, c1 * h1 * w1])
        Q_s = A_s.reshape([b2, c2 * h2 * w2])

        # evaluate normalized similarity matrices (eq 3)
        G_t = torch.mm(Q_t, Q_t.t())
        # G_t = G_t / G_t.norm(p=2)
        G_t = torch.nn.functional.normalize(G_t)

        G_s = torch.mm(Q_s, Q_s.t())
        # G_s = G_s / G_s.norm(p=2)
        G_s = torch.nn.functional.normalize(G_s)

        # calculate the similarity preserving loss (eq 4)
        l_sp = (G_t - G_s).pow(2).mean()

        return l_sp

class SlicedWassersteinDiscrepancy(nn.Module):
    """PyTorch adoption of https://github.com/apple/ml-cvpr2019-swd"""
    def __init__(self, mean=0, sd=1, device='cpu'):
        super(SlicedWassersteinDiscrepancy, self).__init__()
        self.dist = torch.distributions.Normal(mean, sd)
        self.device = device

    def forward(self, p1, p2):
        if p1.shape[1] > 1:
            # For data more than one-dimensional input, perform multiple random
            # projection to 1-D
            proj = self.dist.sample([p1.shape[1], 128]).to(self.device)
            proj *= torch.rsqrt(torch.sum(proj.pow(2), dim=0, keepdim=True))

            p1 = torch.mm(p1, proj)
            p2 = torch.mm(p2, proj)

        p1, _ = torch.sort(p1, 0, descending=True)
        p2, _ = torch.sort(p2, 0, descending=True)

        wdist = (p1 - p2).pow(2).mean()

        return wdist


class RKD(object):
    """
    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.
    relational knowledge distillation.
    arXiv preprint arXiv:1904.05068, 2019.
    """
    def __init__(self, device, eval_dist_loss=True, eval_angle_loss=False):
        super(RKD, self).__init__()
        self.device = device
        self.eval_dist_loss = eval_dist_loss
        self.eval_angle_loss = eval_angle_loss
        self.huber_loss = torch.nn.SmoothL1Loss()

    @staticmethod
    def distance_wise_potential(x):
        x_square = x.pow(2).sum(dim=-1)
        prod = torch.matmul(x, x.t())
        distance = torch.sqrt(
            torch.clamp( torch.unsqueeze(x_square, 1) + torch.unsqueeze(x_square, 0) - 2 * prod,
            min=1e-12))
        mu = torch.sum(distance) / torch.sum(
            torch.where(distance > 0., torch.ones_like(distance),
                        torch.zeros_like(distance)))

        return distance / (mu + 1e-8)

    @staticmethod
    def angle_wise_potential(x):
        e = torch.unsqueeze(x, 0) - torch.unsqueeze(x, 1)
        e_norm = torch.nn.functional.normalize(e, dim=2)
        return torch.matmul(e_norm, torch.transpose(e_norm, -1, -2))

    def eval_loss(self, source, target):

        # Flatten tensors
        source = source.reshape(source.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # normalize
        source = torch.nn.functional.normalize(source, dim=1)
        target = torch.nn.functional.normalize(target, dim=1)

        distance_loss = torch.tensor([0.]).to(self.device)
        angle_loss = torch.tensor([0.]).to(self.device)

        if self.eval_dist_loss:
            distance_loss = self.huber_loss(
                self.distance_wise_potential(source), self.distance_wise_potential(target)
            )

        if self.eval_angle_loss:
            angle_loss = self.huber_loss(
                self.angle_wise_potential(source), self.angle_wise_potential(target)
            )

        return distance_loss, angle_loss
