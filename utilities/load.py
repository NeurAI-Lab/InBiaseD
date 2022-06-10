import os
import torch

path = "/data/output/shruthi.gowda/feature_prior/classification/dml/final/detection/resnet18/inbiased/final_model_old.pth"
model = torch.load(path)
state_dict =  model.state_dict()


torch.save(state_dict, '/data/output/shruthi.gowda/feature_prior/classification/dml/final/detection/resnet18/inbiased/final_model.pth')


import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from evaluation.eval_celebA import CelebA
import torch.utils.data as data_utils
import torch.nn.functional as F
import cv2
import matplotlib as mpl
import os

# baseline_model_path = r'/data/output/fahad.sarfraz/baseline_networks/celeba/baseline-celeba-p-v11-1/checkpoints/WRN-16-8-split[0-0-0]-dropout0.5-celeba-96x96x3.pt'
baseline_model_path = r'/data/output/fahad.sarfraz/baseline_networks/celeba/baseline-celeba-p-v11-0/checkpoints/WRN-16-8-split[0-0-0]-dropout0.5-celeba-96x96x3.pt'
splitnet_model_path = r'/data/output/fahad.sarfraz/splitnet/celeba/splitnet-002-10-celeba-p-v11-r-4/checkpoints/WRN-16-8-split[0-0-2]-dropout0.5-celeba-ind-split-96x96x3.pt'


def get_preds(model, data_loader):
    y = []
    y_pred = []
    is_blond = []

    model.eval()
    for data, label, blond in data_loader:
        data, label, blond = data.cuda(), label.cuda(), blond.cuda()

        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = scores.max(1)

        y.append(label.cpu())
        y_pred.append(predicted.detach().cpu())
        is_blond.append(blond.cpu())

    y = torch.cat(y).numpy()
    y_pred = torch.cat(y_pred).numpy()
    is_blond = torch.cat(is_blond).numpy()

    return y, y_pred, is_blond


# Get the data
dataset = CelebA('/volumes1/datasets/CelebA')
test_dataset = dataset.get_dataset(split='test')
test_loader = data_utils.DataLoader(test_dataset, batch_size=16)
#
# train_dataset = dataset.get_dataset(split='train')
# train_loader = data_utils.DataLoader(train_dataset, batch_size=16)

y = []
y_pred = []
is_blond = []
lst_data = []
for data, label, blond in test_loader:
    y.append(label)
    is_blond.append(blond)
    lst_data.append(data)

y = torch.cat(y).numpy()
is_blond = torch.cat(is_blond).numpy()
data = torch.cat(lst_data).numpy()


baseline_model = torch.load(baseline_model_path).cuda().eval()
splitnet_model = torch.load(splitnet_model_path).cuda().eval()

# Get Model Predictions
b_y, baseline_pred, b_is_blond = get_preds(baseline_model, test_loader)
sn_y, splitnet_pred, sn_is_blond = get_preds(splitnet_model, test_loader)
baseline_iscorrect = baseline_pred == y
splitnet_iscorrect = splitnet_pred == y


# Get indexes for the categories
num_samples = len(y)
blonde_male = [(y[i] == 1) and is_blond[i] for i in range(num_samples)]
blonde_female = [(y[i] == 0) and is_blond[i] for i in range(num_samples)]
non_blonde_male = [(y[i] == 1) and not is_blond[i] for i in range(num_samples)]
non_blonde_female = [(y[i] == 0) and not is_blond[i] for i in range(num_samples)]

# Select Samples
# sel_blond_males = [blonde_male[i] and not baseline_iscorrect[i] and splitnet_iscorrect[i] for i in range(num_samples)]
# sel_nonblond_females = [non_blonde_female[i] and not baseline_iscorrect[i] and splitnet_iscorrect[i] for i in range(num_samples)]

sel_nonblond_males = [bool(non_blonde_male[i]) for i in range(num_samples)]
sel_blond_females = [bool(blonde_female[i]) for i in range(num_samples)]

# =============================================================================
# Add Hooks to the Model
# =============================================================================
num_samples = 20
image_size = (96, 96)
heatmap_threshold = 0.4


interim_features = None
def hook(module, input, output):
    global interim_features
    interim_features = output
    return None


lst_models = [
    (baseline_model, 'Baseline'),
    (splitnet_model, 'splitnet')
]

# test_samples = np.concatenate((data[sel_blond_males], data[sel_nonblond_females]))
# test_labels = np.concatenate((y[sel_blond_males], y[sel_nonblond_females]))

num_samples = 20
test_samples = np.concatenate((data[sel_nonblond_males][:num_samples], data[sel_blond_females][:num_samples]))
test_labels = np.concatenate((y[sel_nonblond_males][:num_samples], y[sel_blond_females][:num_samples]))

test_samples = data[sel_blond_females][:200]
test_labels = y[sel_blond_females][:200]

for model, idt in lst_models:

    out_dir = f'analysis/activations/{idt}_sc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model.residual_block_groups[2].residual_blocks[0].conv2.register_forward_hook(hook)

    for img_idx in range(len(test_samples)):

        sample_data = torch.Tensor(test_samples[img_idx:img_idx + 1])
        sample_label = torch.Tensor(test_labels[img_idx: img_idx + 1])

        # blonde_male_data = data[blonde_male][img_idx:img_idx+1]
        # tensor_image = torch.Tensor(blonde_male_data)
        # labels = torch.Tensor(y[blonde_male][img_idx:img_idx+1])

        # plt.imshow(sample_data[0].permute(1, 2, 0))
        # plt.show()

        pred = model(sample_data.cuda())
        pred = pred[0] if isinstance(pred, tuple) else pred

        # embed = model.relu(model.bn(interim_features))
        # embed = model.pool(model.relu(model.bn(interim_features)))
        # embed = embed.view(-1, model.widened_channels[-1])

        activations = interim_features
        grad = torch.autograd.grad(pred[0, 1], activations, retain_graph=True)[0]
        grad_weight = torch.norm(grad, 2, 1, keepdim=True)

        # Norm Grad
        # act_weight = torch.norm(activations, 2, 1, keepdim=True)
        # saliency = grad_weight * act_weight


        # GradCam
        weights = grad.mean(dim=(2, 3))
        saliency = torch.relu(weights[:, :, None, None] * activations)
        saliency = torch.sum(saliency, dim=1, keepdim=True)

        saliency = F.interpolate(saliency, image_size, mode='bilinear', align_corners=False)
        saliencies_max = torch.amax(saliency, dim=(1, 2))
        saliencies = saliency / saliencies_max[:, None, None]
        # saliencies = torch.mean(torch.stack(saliencies, dim=1), dim=1)
        saliencies = torch.mean(saliencies, dim=1)
        saliency = saliencies.detach().cpu().numpy()


        image = cv2.cvtColor((sample_data[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # image = sample_data[0].permute(1, 2, 0).numpy()
        vmax = np.percentile(saliency, 95)
        normalizer = mpl.colors.Normalize(vmin=saliency.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')

        cam = (mapper.to_rgba(saliency[0, :, :])[:, :, :3] * 255).astype(np.uint8)
        # cam = (mapper.to_rgba(saliency[0, :, :])[:, :, :3]).astype(np.uint8)
        cam = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
        indices = np.where(saliency[0] > heatmap_threshold)
        cam_viz = image.copy()

        # plt.imshow(cam)
        # plt.show()
        #
        # plt.imshow(saliency[0])
        # plt.show()

        cam_viz[indices] = 0.4 * cam[indices] + 0.6 * image[indices]
        # cam_viz = 0.4 * cam + 0.6 * image
        # cam_viz[indices] = 0.3 * saliency[indices] + 0.7 * image[indices]
        # img_saliency = cv2.cvtColor(cam_viz, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f'{out_dir}/{img_idx}.png', cam_viz)