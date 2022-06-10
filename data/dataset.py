import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
from data.data_wrapper import CIFAR10_Wrapper, CIFAR100_Wrapper, STL10_Wrapper, CelebA_Wrapper, MNIST_Wrapper
from data.utils import *
from utilities import dist_utils

# ======================================================================================
# Helper Functions
# ======================================================================================
def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = "/output/feature_prior/classification/dml/v2/imagenet/cache_imgnet.pt"
    # os.path.join(
    #     "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    # )
    cache_path = os.path.expanduser(cache_path)
    return cache_path

class CIFAR10:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        print('==> Preparing CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            #original cifar
            #ds = torchvision.datasets.CIFAR10(root=self.data_path, train=(split == 'train'), download=True, transform=transforms.ToTensor())
            #modified wrapper
            ds = CIFAR10_Wrapper(root=self.data_path, train=False, download=True, transform=transform_test,  transform_fp=transform_test_fp)
        else:
            ds = CIFAR10_Wrapper(root=self.data_path, train=True, download=True, transform=transform_train, transform_fp=transform_train_fp)

        return ds

class CIFAR100:
    """
    CIFAR-100 dataset
    """
    NUM_CLASSES = 100
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        print('==> Preparing CIFAR 100 data..')

        assert split in ['train', 'test']
        if split == 'test':
            #original cifar
            #ds = torchvision.datasets.CIFAR100(root=self.data_path, train=(split == 'train'), download=True, transform=transforms.ToTensor())
            #modified wrapper
            ds = CIFAR100_Wrapper(root=self.data_path, train=False, download=True, transform=transform_test,  transform_fp=transform_test_fp)
        else:
            ds = CIFAR100_Wrapper(root=self.data_path, train=True, download=True, transform=transform_train, transform_fp=transform_train_fp)

        return ds

class STL10():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path, tint=0):
        self.data_path = data_path
        self.tint = tint

    def get_dataset(self, split, transform_train=None, transform_test=None,
                    transform_train_fp=None, transform_test_fp=None):

        if self.tint == 1:
            return torchvision.datasets.STL10(root=self.data_path, split=split, folds=None, download=True,
                                            transform=transforms.ToTensor())
        if split == 'test':
            ds = STL10_Wrapper(root=self.data_path, split=split, folds=None, download=True, transform=transform_test, transform_fp=transform_test_fp)
        else:
            ds = STL10_Wrapper(root=self.data_path, split=split, folds=None, download=True, transform=transform_train, transform_fp=transform_train_fp)

        return ds

class STLTint():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path):
        self.data_path = data_path
        self.stl_dataset = STL10(self.data_path, tint=1)

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        dataset = self.stl_dataset.get_dataset(split)
        if split == 'train':
            ds = stl_tint(dataset, split=split, transform=transform_train, transform_fp=transform_train_fp)
        else:
            ds = stl_tint(dataset, split=split, transform=transform_test, transform_fp=transform_test_fp)

        return ds

class CIFARSmallSub(CIFAR10):
    def get_dataset(self, split, transform_train, transform_test):
        # 1 % is labelled
        assert split in ['train', 'test', 'unlabeled', 'val']
        if split == 'test':
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=transform_train)
        else:
            ds = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=transform_test)
            ds = split_dataset(ds, split,
                               folds=100, num_train_folds=2, num_val_folds=10)
        return ds

class TinyImagenet():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 128

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']
        if split == 'test':
            ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "val_kv_list.txt"),
                               transform=transform_test, transform_fp=transform_test_fp)
        else:
            ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "train_kv_list.txt"),
                               transform=transform_train, transform_fp=transform_train_fp)

        return ds

class Imagenet():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def __init__(self, data_path, cache_dataset):
        self.data_path = data_path
        self.cache_dataset = cache_dataset

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']

        cache_path = _get_cache_path(os.path.join(self.data_path, "train"))
        if self.cache_dataset and os.path.exists(cache_path):
            # Attention, as the transforms are also cached!
            print("Loading dataset_train from {}".format(cache_path))
            ds, _ = torch.load(cache_path)
        else:
            if split == 'test':
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "val"),
                                   transform=transform_test, transform_fp=transform_test_fp)
            else:
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train"),
                                   transform=transform_train, transform_fp=transform_train_fp)

            if self.cache_dataset:
                print("Saving dataset_train to {}".format(cache_path))
                if not os.path.exists(cache_path):
                    dist_utils.mkdir(os.path.dirname(cache_path))
                dist_utils.save_on_master((ds, os.path.join(self.data_path, "train")), cache_path)

        return ds

class Imagenet200():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def __init__(self, data_path, cache_dataset):
        self.data_path = data_path
        self.cache_dataset = cache_dataset

    def get_cache_path(self):
        cache_path = "/output/feature_prior/classification/dml/final/imagenet200/cache_imgnet.pt"
        cache_path = os.path.expanduser(cache_path)
        return cache_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']

        cache_path = self.get_cache_path()
        if self.cache_dataset and os.path.exists(cache_path) and transform_train_fp is not None:
            # Attention, as the transforms are also cached!
            print("Loading dataset_train from {}".format(cache_path))
            ds, _ = torch.load(cache_path)
        else:
            if split == 'test':
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "val"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                                   transform=transform_test, transform_fp=transform_test_fp)
            else:
                ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                                   transform=transform_train, transform_fp=transform_train_fp)

                if self.cache_dataset and transform_train_fp is not None:
                    print("Saving dataset_train to {}".format(cache_path))
                    if not os.path.exists(cache_path):
                        dist_utils.mkdir(os.path.dirname(cache_path))
                    dist_utils.save_on_master((ds, os.path.join(self.data_path, "train")), cache_path)
        return ds

class Imagenet_R():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "annotations.txt"),
                               transform=transform_test, transform_fp=transform_test_fp)

        return ds

class Imagenet_Blurry():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "annotations.txt"),
                               transform=transform_test, transform_fp=transform_test_fp)

        return ds

class Imagenet_A():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "annotations.txt"),
                               transform=transform_test, transform_fp=transform_test_fp, sep=',')

        return ds

class CelebA:
    NUM_CLASSES = 2
    CLASS_NAMES = ['female', 'male']
    MEAN = [0, 0, 0]
    STD = [1, 1, 1, ]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test,
                    transform_train_fp=None, transform_test_fp=None, unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']

        if split == 'test':
            # ds = torchvision.datasets.CelebA(root=self.data_path, split='test', transform=transform_test)
            ds = CelebA_Wrapper(root=self.data_path, split='test', transform=transform_test, transform_fp=transform_test_fp)
        # elif split == 'val':
        #     ds = torchvision.datasets.CelebA(root=self.data_path, split='valid', transform=trans)
        else:
            ds = CelebA_Wrapper(root=self.data_path, split='train', transform=transform_train, transform_fp=transform_train_fp)
        attr_names = ds.attr_names
        attr_names_map = {a: i for i, a in enumerate(attr_names)}
        ds = celeb_indicies(split, ds, attr_names_map, unlabel_skew, transform_train_fp)

        return ds

class Delaunay():
    NUM_CLASSES = 53
    MEAN = [0.5853, 0.5335, 0.4950]
    STD = [0.2348, 0.2260, 0.2242]
    SIZE = 128
    SOBEL_UPSAMPLE_SIZE = 256

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']

        if split == 'test':
            ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "test"),
                               transform=transform_test, transform_fp=transform_test_fp)
        else:
            ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train"),
                                   transform=transform_train, transform_fp=transform_train_fp)
        return ds

class coloredMNIST():
    NUM_CLASSES = 10
    MEAN = [0.1307]
    STD = [0.3081]
    SIZE = 28
    SOBEL_UPSAMPLE_SIZE = 56

    def __init__(self, data_path, color_mode):
        self.data_path = data_path
        self.color_mode = color_mode

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        # dataset = self.stl_dataset.get_dataset(split)
        if split == 'train':
            ds = MNIST_Wrapper(self.data_path, color_mode=self.color_mode, train=True, transform=transform_train, transform_fp=transform_train_fp, download=True)
        else:
            ds = MNIST_Wrapper(self.data_path, color_mode=self.color_mode, train=False, transform=transform_test, transform_fp=transform_test_fp, download=True)

        return ds

class Corrupt_CIFAR10:
    """
    CIFAR-10 dataset
    """
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    NUM_CLASSES = 10
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    SIZE = 32
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        print('==> Preparing Corrupt CIFAR 10 data..')

        assert split in ['train', 'test']
        if split == 'test':
            ds = CIFAR10_Wrapper(root=self.data_path, train=False, download=True, transform=transform_test,  transform_fp=transform_test_fp, corrupt=True)
        else:
            ds = CIFAR10_Wrapper(root=self.data_path, train=True, download=True, transform=transform_train, transform_fp=transform_train_fp, corrupt=True)

        return ds

class Corrupt_TinyImagenet():
    NUM_CLASSES = 10
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 128
    # class_subset  = {149:0, 21:1, 37:2, 137:3, 66:4, 75:5, 111:6, 41:7, 13:8 ,5:9}
    class_subset  = {149:0, 21:1, 37:2, 137:3, 66:4, 75:5, 111:6, 41:7, 13:8 ,5:9}

    def __init__(self, data_path, sev):
        self.data_path = data_path
        self.sev = sev

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']
        if split == 'test':
            ds = ImageFilelist_Corrupt(root=self.data_path, train=False, flist=os.path.join(self.data_path, "val_kv_list.txt"),
                               transform=transform_test, transform_fp=transform_test_fp, class_subset=self.class_subset, sev=self.sev)
        else:
            ds = ImageFilelist_Corrupt(root=self.data_path, train=True, flist=os.path.join(self.data_path, "train_kv_list.txt"),
                               transform=transform_train, transform_fp=transform_train_fp, class_subset=self.class_subset, sev=self.sev)

        return ds

class Stylized_TinyImagenet():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 128

    def __init__(self, data_path, data_path_style):
        self.data_path = data_path
        self.data_path_style = data_path_style

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        assert split in ['train', 'test']
        if split == 'test':
            ds = ImageFilelist(root=self.data_path, flist=os.path.join(self.data_path, "val_kv_list.txt"),
                               transform=transform_test, transform_fp=transform_test_fp)
        else:
            ds = ImageFilelist_style(root=self.data_path, root_style=self.data_path_style, flist=os.path.join(self.data_path, "train_kv_list.txt"),
                               transform=transform_train, transform_fp=transform_train_fp)

        return ds

class Stylized_STL10():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path):
        self.data_path = data_path
        self.stl_dataset = STL10(self.data_path, tint=1)

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        dataset = self.stl_dataset.get_dataset(split)
        if split == 'train':
            ds = stl_style(dataset, split=split, transform=transform_train, transform_fp=transform_train_fp, style=True)
        else:
            ds = stl_style(dataset, split=split, transform=transform_test, transform_fp=transform_test_fp, style=False)

        return ds

class Stylized_STLTint():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path):
        self.data_path = data_path
        self.stl_dataset = STL10(self.data_path, tint=1)

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        dataset = self.stl_dataset.get_dataset(split)
        if split == 'train':
            ds = stl_tint(dataset, split=split, transform=transform_train, transform_fp=transform_train_fp, style=True)
        else:
            ds = stl_tint(dataset, split=split, transform=transform_test, transform_fp=transform_test_fp, style=False)

        return ds

class Stylized_CelebA:
    NUM_CLASSES = 2
    CLASS_NAMES = ['female', 'male']
    MEAN = [0, 0, 0]
    STD = [1, 1, 1, ]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self, data_path):
        self.data_path = data_path

    def get_dataset(self, split, transform_train, transform_test,
                    transform_train_fp=None, transform_test_fp=None, unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']

        dataset = torchvision.datasets.CelebA(root=self.data_path, split=split, transform=transform_train)
        # dataset = CelebA_Wrapper(root=self.data_path, split=split, transform=transforms.ToTensor(), transform_fp=None)
        ds = dataset
        attr_names = ds.attr_names
        attr_names_map = {a: i for i, a in enumerate(attr_names)}
        ds = celeb_indicies(split, ds, attr_names_map, unlabel_skew, transform_train_fp)

        trans_style = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        if split == 'test':
            ds = celeb_style(ds, split='train', transform=trans_style, transform_fp=transform_train_fp, style=False)
        else:
            ds = celeb_style(ds, split='train', transform=trans_style, transform_fp=transform_train_fp, style=True)

        return ds

class Stylized_CMNIST():
    NUM_CLASSES = 10
    MEAN = [0.1307]
    STD = [0.3081]
    SIZE = 28
    SOBEL_UPSAMPLE_SIZE = 56

    def __init__(self, data_path):
        self.data_path = data_path
        self.color_mode = 'fg'

    def get_dataset(self, split, transform_train, transform_test, transform_train_fp=None, transform_test_fp=None):
        ds = MNIST_Wrapper(self.data_path, color_mode=self.color_mode, train=True, transform=transforms.ToTensor(),
                          download=True)

        trans_style = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        if split == 'train':
            ds = cmnist_style(ds, split='train', transform=trans_style, style=True)
        else:
            ds = cmnist_style(ds, split='test', transform=trans_style, style=False )

        return ds


DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'cifarsmallsub': CIFARSmallSub,
    'tinyimagenet': TinyImagenet,
    'stl10': STL10,
    'stl_fourier' : STL10,
    'stltint': STLTint,
    'imagenet': Imagenet,
    'imagenet200': Imagenet200,
    'imagenet_r': Imagenet_R,
    'imagenet_blurry': Imagenet_Blurry,
    'imagenet_a': Imagenet_A,
    'tinyimagenet_fourier' : TinyImagenet,
    'celeba' : CelebA,
    'delaunay': Delaunay,
    'col_mnist': coloredMNIST,
    'cor_cifar10': Corrupt_CIFAR10,
    'cor_tinyimagenet':Corrupt_TinyImagenet,
    'style_tiny': Stylized_TinyImagenet,
    'style_stltint': Stylized_STLTint,
    'style_stl10': Stylized_STL10,
    'style_celeba': Stylized_CelebA,
    'style_cmnist': Stylized_CMNIST,

}
