import cv2
import numpy as np
import PIL
from PIL import Image
from data.utils import plot
import torchvision.transforms as transforms
from imagecorruptions import get_corruption_names, corrupt

SAVE_SOBEL = False

class ImageCorruptions:
    def __init__(self, args):
        self.severity = args.corrupt_severity
        self.corruption_name = args.corrupt_name

    def __call__(self, image, labels=None):

        image = np.array(image)
        cor_image = corrupt(image, corruption_name=self.corruption_name,
                        severity=self.severity)

        return Image.fromarray(cor_image)

class transform_canny_edge(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = Image.fromarray(cv2.bilateralFilter(np.array(img),5,75,75))
        gray_scale = transforms.Grayscale(1)
        image = gray_scale(img)
        edges = cv2.Canny(np.array(image), 100, 200)
        out = np.stack([edges, edges, edges], axis=-1)
        to_pil = transforms.ToPILImage()
        out = to_pil(out)
        return out

class transform_sobel_edge(object):
    def __init__(self, args, upsample_size=0):
        self.gauss_ksize = args.sobel_gauss_ksize
        self.sobel_ksize = args.sobel_ksize
        self.upsample = args.sobel_upsample
        self.upsample_size = upsample_size

    def __call__(self, img, boxes=None, labels=None):

        if SAVE_SOBEL:
            plot(img, 'stl_before_sobel')

        if self.upsample == 'True':
            curr_size = img.size[0]
            resize_up = transforms.Resize(max(curr_size, self.upsample_size), 3)
            resize_down = transforms.Resize(curr_size, 3)
            rgb = np.array(resize_up(img))
        else:
            rgb = np.array(img)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb = cv2.GaussianBlur(rgb, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)
        sobelx = cv2.Sobel(rgb, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        imgx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(rgb, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        imgy = cv2.convertScaleAbs(sobely)
        tot = np.sqrt(np.square(sobelx) + np.square(sobely))
        imgtot = cv2.convertScaleAbs(tot)
        sobel_img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))

        sobel_img = resize_down(sobel_img) if self.upsample == 'True' else sobel_img

        if SAVE_SOBEL:
            plot(sobel_img, 'stl_sobel')

        return sobel_img

class transform_lowpass_fft(object):

    def __init__(self, args, size):
        self.args = args
        self.size = size
        #self.radius = args.radius

    def __call__(self, img):
        if SAVE_SOBEL:
            plot(img, 'before_fourier')

        r = 4 #self.radius  # how narrower the window is
        ham = np.hamming(self.size)[:, None]  # 1D hamming
        ham2d = np.sqrt(np.dot(ham, ham.T)) ** r  # expand to 2D hamming

        gray_image = np.array(img)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        f = cv2.dft(gray_image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
        f_filtered = ham2d * f_complex

        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img * 255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)
        fourier_img = Image.fromarray(cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR))

        if SAVE_SOBEL:
            plot(fourier_img, 'fourier')

        return fourier_img

def transform_cifar(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        train_aug =[
                #transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(ds_class.SIZE, padding = 4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(2),
                transforms.ToTensor(),
            ]
        if args.norm_std == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug =[
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
                #normalize,
            ]
        if args.norm_std == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    else:
        train_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.RandomCrop(ds_class.SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(2),
                transforms.ToTensor(),
            ]
        if args.norm_fp == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        if args.norm_fp == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_stl(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        train_aug = [
                transforms.RandomCrop(ds_class.SIZE, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize if args.norm_aug else None
            ]
        if args.norm_std == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug = [
                #transforms.Resize(ds_class.SIZE),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
                #normalize,
            ]
        if args.norm_std == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)
    else:
        train_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.RandomCrop(ds_class.SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(2),
                transforms.ToTensor(),

            ]
        if args.norm_fp == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        if args.norm_fp == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_stltint(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(ds_class.SIZE, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize,
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
                #normalize,
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.RandomCrop(ds_class.SIZE, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(2),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToPILImage(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                #transforms.Resize(size),
                #transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

    return transform_train, transform_test

def transform_imagenet(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((ds_class.SIZE,ds_class.SIZE)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]
        )
    elif args.ft_prior == 'sobel':
        transform_train = transforms.Compose(
            [
                transform_sobel_edge(args),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
            ]
        )
        transform_test = transforms.Compose(
            [
                transform_sobel_edge(args),
                transforms.Resize((ds_class.SIZE,ds_class.SIZE)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ]
        )

    return transform_train, transform_test

def transform_tinyimagenet(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        train_aug = [
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize,
            ]
        if args.norm_std == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)
        test_aug = [
                transforms.Resize(ds_class.SIZE),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                # normalize,
            ]
        if args.norm_std == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)
    elif args.ft_prior == 'sobel':
        train_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #normalize,
            ]
        if args.norm_std == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)
        test_aug = [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.Resize(ds_class.SIZE),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                # normalize,
            ]
        if args.norm_std == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_fourier_tinyimagenet(args, ds_class):

    if args.ft_prior == 'std':
        transform_test = transforms.Compose(
            [
                transforms.Resize(ds_class.SIZE),
                transform_lowpass_fft(args, ds_class.SIZE),
                transforms.ToTensor(),
                # normalize,
            ]
        )
    else:
        transform_test = transforms.Compose(
            [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.Resize(ds_class.SIZE),
                transform_lowpass_fft(args, ds_class.SIZE),
                transforms.ToTensor(),
                # normalize,
            ]
        )

    return None, transform_test

def transform_celeba(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        transform_train = transforms.Compose(
            [
                transforms.Resize((ds_class.SIZE, ds_class.SIZE)),
                transforms.ToTensor(),
                #normalize,
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((ds_class.SIZE, ds_class.SIZE)),
                transforms.ToTensor(),
                #normalize,
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.Resize((ds_class.SIZE, ds_class.SIZE)),
                transforms.ToTensor(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.Resize((ds_class.SIZE, ds_class.SIZE)),
                transforms.ToTensor(),
            ]
        )

    return transform_train, transform_test

def transform_mnist(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        train_aug =[
                transforms.ToTensor(),
            ]
        if args.norm_std == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug =[
                transforms.ToTensor(),
            ]
        if args.norm_std == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    else:
        train_aug = [
                #transform_canny_edge(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.ToTensor(),
            ]
        if args.norm_fp == 'True':
            train_aug.append(normalize)
        transform_train = transforms.Compose(train_aug)

        test_aug = [
                #transform_canny_edge(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.ToTensor(),
            ]
        if args.norm_fp == 'True':
            test_aug.append(normalize)
        transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test

def transform_mnist_style(args, ds_class):

    normalize = transforms.Normalize(mean=ds_class.MEAN,
                                     std=ds_class.STD)

    if args.ft_prior == 'std':
        train_aug =[
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        transform_train = transforms.Compose(train_aug)

        test_aug =[
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        transform_test = transforms.Compose(test_aug)

    else:
        train_aug = [
                #transform_canny_edge(),
                transforms.ToPILImage(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.ToTensor(),
            ]
        transform_train = transforms.Compose(train_aug)

        test_aug = [
                #transform_canny_edge(),
                transforms.ToPILImage(),
                transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
                transforms.ToTensor(),
            ]
        transform_test = transforms.Compose(test_aug)

    return transform_train, transform_test


def build_transforms(args, ds_class):

    if args.dataset == "cifar10" or args.dataset == 'cor_cifar10' or args.dataset == 'cifar100' or args.dataset == 'cifarsmallsub':
        return transform_cifar(args, ds_class)
    elif args.dataset == "stl10":
        return transform_stl(args, ds_class)
    elif args.dataset == "stltint" or args.dataset=="style_stltint" or args.dataset == 'style_stl10':
        return transform_stltint(args, ds_class)
    elif args.dataset == "imagenet" or args.dataset == "imagenet200":
        return transform_imagenet(args, ds_class)
    elif args.dataset == "tinyimagenet" or args.dataset == "imagenet_r" or args.dataset == "imagenet_blurry" or args.dataset == "imagenet_a"  \
            or args.dataset == 'cor_tinyimagenet' or args.dataset == 'style_tiny':
        return transform_tinyimagenet(args, ds_class)
    elif args.dataset == "tinyimagenet_fourier":
        return transform_fourier_tinyimagenet(args, ds_class)
    elif args.dataset == "stl_fourier":
        return transform_fourier_tinyimagenet(args, ds_class)
    elif args.dataset == "celeba" or args.dataset == 'style_celeba':
        return transform_celeba(args, ds_class)
    elif args.dataset == "col_mnist":
        return transform_mnist(args, ds_class)
    elif args.dataset == "style_cmnist":
        return transform_mnist_style(args, ds_class)
