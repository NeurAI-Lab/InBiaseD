import torch
from data.dataset import DATASETS
import torchvision.transforms as transforms
from data.transforms import transform_sobel_edge

def apply_sobel(args, x):

    ds_class = DATASETS[args.dataset](args.data_path)

    transform_test = transforms.Compose(
        [
            transforms.ToPILImage(),
            transform_sobel_edge(args, ds_class.SOBEL_UPSAMPLE_SIZE),
            # transforms.Resize(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )

    x_sob_lst = []
    for i in range(x.shape[0]):
        x_new = x[i].squeeze() #.detach().cpu().numpy()  # remove batch dimension # B X C H X W ==> C X H X W
        x_sob_lst.append(transform_test(x_new))

    x_sob = torch.stack(x_sob_lst)
    return x_sob.to('cuda' if args.cuda else 'cpu')