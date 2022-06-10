import os
from torchvision import datasets, transforms

# def get_test_dataset(dataset_choice, dataset_path='dataset', remove_norm=False):
#
#     if dataset_choice == 'imagenet_r':
#         normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
#                                          std=[0.2302, 0.2265, 0.2262])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "annotations.txt"),
#             transform=transforms.Compose([
#                 transforms.Resize(64),
#                 transforms.CenterCrop(56),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
#         )
#
#     elif dataset_choice == 'imagenet_blurry':
#         normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
#                                          std=[0.2302, 0.2265, 0.2262])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "annotations.txt"),
#             transform=transforms.Compose([
#                 transforms.Resize(64),
#                 transforms.CenterCrop(56),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
#         )
#
#     elif dataset_choice == 'imagenet_a':
#         normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
#                                          std=[0.2302, 0.2265, 0.2262])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "annotations.txt"),
#             transform=transforms.Compose([
#                 transforms.Resize(64),
#                 transforms.CenterCrop(56),
#                 transforms.ToTensor(),
#                 normalize,
#             ]),
#             sep=','
#         )
#
#     elif dataset_choice == 'celeba':
#         celeba_dataset = CelebA(dataset_path)
#         dataset = celeba_dataset.get_dataset(split='test')
#
#     elif dataset_choice == 'stl_tinted':
#         STL_TEST_TRANSFORMS = transforms.Compose([
#             transforms.ToPILImage(),
#             # transforms.CenterCrop(96),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 (0.4192, 0.4124, 0.3804),
#                 (0.2714, 0.2679, 0.2771)
#             )
#         ])
#
#         dataset = STLTint(dataset_path, 'test', STL_TEST_TRANSFORMS)
#
#     elif dataset_choice == 'domain_net_real':
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "real_test.txt"),
#             transform=transform
#             )
#
#     elif dataset_choice == 'domain_net_clipart':
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "clipart_test.txt"),
#             transform=transform
#             )
#
#     elif dataset_choice == 'domain_net_infograph':
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "infograph_test.txt"),
#             transform=transform
#             )
#
#     elif dataset_choice == 'domain_net_painting':
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "painting_test.txt"),
#             transform=transform
#             )
#
#     elif dataset_choice == 'domain_net_sketch':
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         dataset = ImageFilelist(
#             root=dataset_path,
#             flist=os.path.join(dataset_path, "sketch_test.txt"),
#             transform=transform
#             )
#
#
#     else:
#         raise ValueError(f'{dataset_choice} not supported')
#
#     return dataset