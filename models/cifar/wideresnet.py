import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

        self.no_bnrelu = False

    def forward(self, x):
        if self.no_bnrelu:
            if not self.equalInOut:
                x = x
            else:
                out = x
        else:
            if not self.equalInOut:
                x = self.relu1(self.bn1(x))
            else:
                out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x, no_bnrelu=False):
        if no_bnrelu:
            self.layer[0].no_bnrelu = True
            res = self.layer(x)
            self.layer[0].no_bnrelu = False
            return res
        else:
            return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        # required for making creating consistent spatial dimension for FSP
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        # save intermediate results as
        self.res0 = self.conv1(x)
        self.pool0 = self.res0

        self.res1 = self.block1(self.res0)
        self.pool1 = self.max_pool1(self.res1)

        self.res2 = self.block2(self.res1)
        self.pool2 = self.max_pool2(self.res2)

        self.res3 = self.block3(self.res2)

        out = self.relu(self.bn1(self.res3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class ONE_WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0):
        super(ONE_WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        ## Low level feature blocks
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)

        # High level feature blocks for branching
        # 3rd block
        self.block3_1 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.block3_2 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.block3_3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # Gate Block
        self.avgpool_c = nn.AvgPool2d(16)
        self.control_v1 = nn.Linear(nChannels[2], 3)
        self.bn_v1 = nn.BatchNorm1d(3)

        # global average pooling and classifier
        self.avgpool = nn.AvgPool2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.nChannels = nChannels[3]

        # classifier branches
        self.classifier1 = nn.Linear(nChannels[3], num_classes)
        self.classifier2 = nn.Linear(nChannels[3], num_classes)
        self.classifier3 = nn.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        # extract low level features
        low_feat = self.conv1(x)
        low_feat = self.block1(low_feat)
        low_feat = self.block2(low_feat)

        # Gate
        out_gate = self.avgpool_c(low_feat)
        out_gate = out_gate.view(out_gate.size(0), -1)
        out_gate = self.control_v1(out_gate)
        out_gate = self.bn_v1(out_gate)
        out_gate = F.relu(out_gate)
        out_gate = F.softmax(out_gate, dim=1)

        # high level features from the three branches
        high_feat_1 = self.block3_1(low_feat)
        high_feat_1 = self.relu(self.bn1(high_feat_1))

        high_feat_2 = self.block3_2(low_feat)
        high_feat_2 = self.relu(self.bn1(high_feat_2))

        high_feat_3 = self.block3_3(low_feat)
        high_feat_3 = self.relu(self.bn1(high_feat_3))

        # Flatten the features
        input_c1 = F.avg_pool2d(high_feat_1, 8)
        input_c1 = input_c1.view(-1, self.nChannels)

        input_c2 = F.avg_pool2d(high_feat_2, 8)
        input_c2 = input_c2.view(-1, self.nChannels)

        input_c3 = F.avg_pool2d(high_feat_3, 8)
        input_c3 = input_c3.view(-1, self.nChannels)

        # Get classification outputs
        out_c1 = self.classifier1(input_c1)
        out_c2 = self.classifier2(input_c2)
        out_c3 = self.classifier3(input_c3)

        # Gate additions
        weight_c1 = out_gate[:, 0].repeat(out_c1.size()[1], 1).transpose(0,1)
        weight_c2 = out_gate[:, 1].repeat(out_c2.size()[1], 1).transpose(0,1)
        weight_c3 = out_gate[:, 2].repeat(out_c3.size()[1], 1).transpose(0,1)

        # Teacher Output
        out_t = weight_c1 * out_c1 + weight_c2 * out_c2 + weight_c3 * out_c3
        # out_t = (out_c1 + out_c2 + out_c3) / 3

        return out_c1, out_c2, out_c3, out_t


def wrn(**kwargs):
    """
    Constructs a ResNet model.
    """
    return WideResNet(**kwargs)

def one_wrn(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ONE_WideResNet(**kwargs)


if __name__ == '__main__':
    from torchsummary import summary

    model = ONE_WideResNet(16, 2, 10)
    print(model)

    model = model.to('cuda')
    summary(model, input_size=(3, 32, 32))
