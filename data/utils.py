import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data as data_utils
from torch.utils.data import Dataset, TensorDataset, DataLoader
from third_party.util import stylize
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
try:
    from imagecorruptions import corrupt
except ImportError:
    print("Import Error corruption")

SAVE_IMG = False

VALID_SPURIOUS = [
    'TINT',  # apply a fixed class-wise tinting (meant to not affect shape)
]

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            imlist.append((impath, int(imlabel)))

    return imlist

def path_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            items = line.strip().split(sep)
            impath, imlabel = items
            imlist.append((os.path.join(root, impath), int(imlabel)))

    return imlist

def pathstyle_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            items = line.strip().split(sep)
            impath, imlabel = items
            impath = os.path.splitext(impath)[0] + '.jpg'
            imlist.append((os.path.join(root, impath), int(imlabel)))

    return imlist

def subset_flist_reader(flist, sep, class_list):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            if int(imlabel) in class_list.keys():
                imlist.append((impath, int(imlabel)))

    return imlist

def style_flist_reader(root, flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)

            if impath[0] == '.':
                impath = impath[impath.find('val'):]

            if os.path.exists(os.path.join(root, impath)):
                imlist.append((impath, int(imlabel)))

    return imlist

def folder_reader(data_dir):
    all_img_files = []
    all_labels = []

    class_names = os.walk(data_dir).__next__()[1]
    for index, class_name in enumerate(class_names):
        label = index
        img_dir = os.path.join(data_dir, class_name)
        img_files = os.walk(img_dir).__next__()[2]

        for img_file in img_files:
            img_file = os.path.join(img_dir, img_file)
            img = Image.open(img_file)
            if img is not None:
                all_img_files.append(img_file)
                all_labels.append(int(label))

    return all_img_files, all_labels

def subset_folder_reader(data_dir, flist):
    all_img_files = []
    all_labels = []

    with open(flist, 'r') as rf:
        for line in rf.readlines():
            imfolder, imlabel = line.strip().split(' ')
            class_name = imfolder
            label = imlabel
            img_dir = os.path.join(data_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(int(label))

    return all_img_files, all_labels

class ImageFilelist(torch.utils.data.Dataset):
    def __init__(self, root, flist=None, folderlist=None, subset_folderlist=None, transform=None, transform_fp=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = []
        if flist:
            self.imlist = flist_reader(flist,sep)
            #self.imlist = style_flist_reader(root, flist,sep)

        elif subset_folderlist:
            self.images, self.labels = subset_folder_reader(folderlist, subset_folderlist)
        else:
            self.images, self.labels = folder_reader(folderlist)

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        if self.imlist:
            impath, target = self.imlist[index]
            img = self.loader(os.path.join(self.root, impath))
        else:
            img = self.loader(self.images[index])
            target = self.labels[index]

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.imlist) if self.imlist else len(self.images)

class ImageFilelist_Corrupt(torch.utils.data.Dataset):
    def __init__(self, root, train=True, flist=None, transform=None, transform_fp=None, target_transform=None,
                 flist_reader=subset_flist_reader, loader=default_loader, sep=' ', class_subset={}, sev=1):
        self.root = root
        self.train = train
        self.imlist = []
        self.imlist = flist_reader(flist, sep, class_subset)
        self.class_subset = class_subset

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.loader = loader
        self.corrupt_list = ['brightness', 'contrast', 'gaussian_noise', 'frost', 'elastic_transform',
                        'gaussian_blur', 'defocus_blur', 'impulse_noise', 'saturate', 'pixelate']
        self.sev = sev

    def __getitem__(self, index):

        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        target = self.class_subset[target]

        if self.train:
            img = corrupt(np.asarray(img), corruption_name=self.corrupt_list[target], severity=self.sev)
        else:
            rand_target = random.randint(0, 9)
            img = corrupt(np.asarray(img), corruption_name=self.corrupt_list[rand_target], severity=self.sev)

        img = Image.fromarray(img)
        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)

class ImageFilelist_style(torch.utils.data.Dataset):
    def __init__(self, root, root_style, flist=None,transform=None, transform_fp=None, target_transform=None,
                 flist_reader=path_flist_reader, flist_reader_style=pathstyle_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.root_style = root_style
        self.imlist = []
        self.imlist_orig = flist_reader(root, flist,sep)
        self.imlist_style = flist_reader_style(root_style, flist,sep)

        self.imlist.extend(self.imlist_orig)
        self.imlist.extend(self.imlist_style)

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        if self.imlist:
            impath, target = self.imlist[index]
            img = self.loader(impath)
        else:
            img = self.loader(self.images[index])
            target = self.labels[index]

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)

def add_spurious(ds, mode):
    assert mode in VALID_SPURIOUS

    loader = DataLoader(ds, batch_size=32, num_workers=1,
                        pin_memory=False, shuffle=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs)
    ys = torch.cat(ys)

    colors = torch.tensor([(2, 1, 0), (1, 2, 0), (1, 1, 0),
                           (0, 2, 1), (0, 1, 2), (0, 1, 1),
                           (1, 0, 2), (2, 0, 1), (1, 0, 1),
                           (1, 1, 1)])

    colors = colors / torch.sum(colors + 0.0, dim=1, keepdim=True)

    xs_tint = (xs + colors[ys].unsqueeze(-1).unsqueeze(-1) / 3).clamp(0, 1)

    return TensorDataset(xs_tint, ys)

class stl_tint(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None, transform_fp=None, target_transform=None, style=False):
        self.dataset = add_spurious(dataset, 'TINT') if split =='train' else dataset
        self.dataset_orig = dataset

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.style = style
        if self.style:
            self.style_dataset = stylize(self.dataset)

    def __getitem__(self, index):

        if self.style:
            img, target = self.style_dataset[index]
        else:
            img, target = self.dataset[index]

        #torchvision.utils.save_image(img,'/volumes2/feature_prior_project/images/stltint_style/{}.jpg'.format(target))

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.style_dataset) if self.style else len(self.dataset)

class stl_style(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None, transform_fp=None, target_transform=None, style=True):
        self.dataset = dataset
        self.dataset_orig = dataset

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.style = style
        if self.style:
            self.style_dataset = stylize(self.dataset)

    def __getitem__(self, index):

        if self.style:
            img, target = self.style_dataset[index]
        else:
            img, target = self.dataset[index]

        # torchvision.utils.save_image(img,'/volumes2/feature_prior_project/images/stl10_style/{}.jpg'.format(target))

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.style_dataset) if self.style else len(self.dataset)

class celeb_style(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None, transform_fp=None, target_transform=None, style=True):
        self.dataset = dataset

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.style = style
        if self.style:
            self.style_dataset = stylize(self.dataset)

    def __getitem__(self, index):

        if self.style:
            img, target = self.style_dataset[index]
        else:
            img, target = self.dataset[index]

        # torchvision.utils.save_image(img,'/volumes2/feature_prior_project/images/celeb_style/{}.jpg'.format(target))

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.style_dataset) if self.style else len(self.dataset)

class cmnist_style(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None, transform_fp=None, target_transform=None, style=True):
        self.dataset = dataset

        self.transform = transform
        self.transform_fp = transform_fp
        self.target_transform = target_transform
        self.style = style
        if self.style:
            self.style_dataset = stylize(self.dataset)

    def __getitem__(self, index):

        if self.style:
            img, target = self.style_dataset[index]
        else:
            img, target = self.dataset[index]

        # torchvision.utils.save_image(img,'/volumes2/feature_prior_project/images/col_mnist/bg/{}.jpg'.format(target))

        img_fp = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform_fp is not None:
            img_fp = self.transform_fp(img_fp)
            return img, img_fp, target
        else:
            return img, target

    def __len__(self):
        return len(self.style_dataset) if self.style else len(self.dataset)

def split_dataset(ds, split, folds=10, num_train_folds=2, num_val_folds=0):
    all_data_points = np.arange(len(ds))
    every_other = all_data_points[::folds]
    train_folds = num_train_folds
    val_folds = num_val_folds
    train_points = np.concatenate([every_other + i
                                    for i in range(0, train_folds)])
    if num_val_folds > 0:
        val_points = np.concatenate([every_other + i
                                       for i in range(train_folds,
                                                      train_folds + val_folds)])
    if folds - (train_folds + val_folds) > 0:
        unlabelled_points = np.concatenate([every_other + i
                                            for i in range(train_folds + val_folds,
                                                        folds)])
    if split == 'train':
        ds = torch.utils.data.Subset(ds, train_points)
    elif split.startswith('val'):
        if num_val_folds == 0:
            raise ValueError("Can't create a val set with 0 folds")
        ds = torch.utils.data.Subset(ds, val_points)
    else:
        if folds - (train_folds + val_folds) == 0:
            raise ValueError('no room for unlabelled points')
        ds = torch.utils.data.Subset(ds, unlabelled_points)
    return ds

def plot(img, name):
    dir = "/volumes2/feature_prior_project/art/feature_prior/vis"
    out = rf"{dir}/{name}.jpg"
    img.save(out)

def celeb_indicies(split, ds, attr_names_map, unlabel_skew=True, transform_train_fp=None, transform_test_fp=None):
    male_mask = ds.attr[:, attr_names_map['Male']] == 1
    female_mask = ds.attr[:, attr_names_map['Male']] == 0
    blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 1
    not_blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 0

    indices = torch.arange(len(ds))

    if split == 'train':
        male_blond = indices[torch.logical_and(male_mask, blond_mask)]
        male_not_blond = indices[torch.logical_and(male_mask, not_blond_mask)]
        female_blond = indices[torch.logical_and(female_mask, blond_mask)]
        female_not_blond = indices[torch.logical_and(female_mask, not_blond_mask)]
        p_male_blond = len(male_blond) / float(len(indices))
        p_male_not_blond = len(male_not_blond) / float(len(indices))
        p_female_blond = len(female_blond) / float(len(indices))
        p_female_not_blond = len(female_not_blond) / float(len(indices))

        # training set must have 500 male_not_blond and 500 female_blond
        train_N = 1000
        training_male_not_blond = male_not_blond[:train_N]
        training_female_blond = female_blond[:train_N]

        unlabeled_male_not_blond = male_not_blond[train_N:]
        unlabeled_female_blond = female_blond[train_N:]
        unlabeled_male_blond = male_blond
        unlabeled_female_not_blond = female_not_blond

        if unlabel_skew:
            # take 1000 from each category
            unlabeled_N = 1000
            unlabeled_male_not_blond = unlabeled_male_not_blond[:unlabeled_N]
            unlabeled_female_blond = unlabeled_female_blond[:unlabeled_N]
            unlabeled_male_blond = unlabeled_male_blond[:unlabeled_N]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:unlabeled_N]
        else:
            total_N = 4000
            extra = total_N - int(p_male_not_blond * total_N) - int(p_female_blond * total_N) - int(
                p_male_blond * total_N) - int(p_female_not_blond * total_N)
            unlabeled_male_not_blond = unlabeled_male_not_blond[:int(p_male_not_blond * total_N)]
            unlabeled_female_blond = unlabeled_female_blond[:int(p_female_blond * total_N)]
            unlabeled_male_blond = unlabeled_male_blond[:int(p_male_blond * total_N)]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:(int(p_female_not_blond * total_N) + extra)]

        train_indices = np.concatenate([training_male_not_blond, training_female_blond])
        unlabelled_indices = np.concatenate(
            [unlabeled_male_not_blond, unlabeled_female_blond, unlabeled_male_blond, unlabeled_female_not_blond])
        for index in unlabelled_indices:
            assert index not in train_indices

        if split == 'train':
            indices = train_indices
        else:
            indices = unlabelled_indices

    imgs = []
    imgs_fp = []
    ys = []
    metas = []
    is_blonde= []
    for i in indices:

        if transform_train_fp is not None:
            img, img_fp, attr = ds[i]
            imgs_fp.append(img_fp)
        else:
            img, attr = ds[i]

        imgs.append(img)
        ys.append(attr[attr_names_map['Male']])
        is_blonde.append(blond_mask[i])
        if male_mask[i]:
            agree = False if blond_mask[i] else True
        else:
            agree = True if blond_mask[i] else False
        metas.append({'agrees': agree, 'blond': blond_mask[i], 'male': male_mask[i]})

    print(split, len(indices))
    if split == 'test':
        if transform_train_fp is not None:
            return data_utils.TensorDataset(torch.stack(imgs), torch.stack(imgs_fp), torch.tensor(ys))
        else:
            return data_utils.TensorDataset(torch.stack(imgs), torch.tensor(ys))#, torch.tensor(is_blonde))
    else:
        if transform_train_fp is not None:
            return data_utils.TensorDataset(torch.stack(imgs), torch.stack(imgs_fp), torch.tensor(ys))
        else:
            return data_utils.TensorDataset(torch.stack(imgs), torch.tensor(ys))

