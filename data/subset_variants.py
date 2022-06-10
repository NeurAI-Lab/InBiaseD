import os
import numpy as np
from glob import glob

# get tiny imagenet classes
tinyimg_path = r'/data/input/datasets/tiny_imagenet/tiny-imagenet-200/train_kv_list.txt'

imlist = []
class_to_idx = {}
with open(tinyimg_path, 'r') as rf:
    for line in rf.readlines():
        impath, imlabel = line.strip().split()
        class_to_idx[impath.split('/')[1]] = int(imlabel)
        imlist.append((impath, int(imlabel)))

# =============================================================================
# Stylized Images
# =============================================================================
style_path = r'/data/input/datasets/Stylized-Imagenet/val'
lst_classes = os.listdir(style_path)

lst_classes = [label.split('_')[0] for label in lst_classes if not label.endswith('.txt')]

lst_common_classes = [label for label in lst_classes if label in class_to_idx]
print(f'Common Classes: {len(lst_common_classes)}')

# Create Annotation File
lst_images = glob(style_path + '/*/*.png')

with open(os.path.join(style_path, 'annotations.txt'), 'w') as f:
    for img in lst_images:
        rel_path = os.path.relpath(img, style_path)
        img_class = rel_path.split('/')[0].split('_')[0]

        if img_class in lst_common_classes:
            label_index = class_to_idx[img_class]
            f.write(f'{rel_path} {label_index}\n')

        else:
            print(f'{rel_path} not Covered')



