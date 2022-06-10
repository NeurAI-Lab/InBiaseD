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
    def __init__(self, depth, width=1, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * width, 32 * width, 64 * width]
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
        # out = F.dropout(out, 0.5)
        return self.fc(out)


class wrapper_model(nn.Module):
    def __init__(self, model, inference_branch):
        super(wrapper_model, self).__init__()
        self._model = model
        self.inference_branch = inference_branch

    def forward(self, x):
        out = self._model(x)[self.inference_branch]
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = WideResNet(16, 2)
    print(model)

    model = model.to('cuda')
    summary(model, input_size=(3, 32, 32))
