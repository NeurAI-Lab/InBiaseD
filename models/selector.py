from .mlp import mlp
from .wideresnet import WideResNet
from .resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def select_model(arch, num_classes, cifar_resnet=False):
    if arch == 'WRN-40-2':
        model = WideResNet(40, 2, num_classes)

    elif arch == 'WRN-28-2':
        model = WideResNet(28, 2, num_classes)

    elif arch == 'WRN-16-2':
        model = WideResNet(16, 2, num_classes)

    elif arch == 'WRN-10-2':
        model = WideResNet(10, 2, num_classes)

    elif arch == 'WRN-28-10':
        model = WideResNet(28, 10, num_classes)

    elif arch == 'WRN-34-10':
        model = WideResNet(34, 10, num_classes)

    elif arch == 'ResNet10':
        model = ResNet10(num_classes)

    elif arch == 'ResNet18':
        model = ResNet18(num_classes, cifar_resnet)

    elif arch == 'ResNet34':
        model = ResNet34(num_classes)

    elif arch == 'ResNet50':
        model = ResNet50(num_classes)

    elif arch == 'ResNet101':
        model = ResNet101(num_classes)

    elif arch == 'ResNet152':
        model = ResNet152(num_classes)

    elif arch == 'MLP':
        model = mlp(num_classes)

    else:
        raise NotImplemented

    return model
