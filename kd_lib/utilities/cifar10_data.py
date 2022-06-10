"""
cifar-10 dataset, with support for random labels
"""
import numpy as np
import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels

    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]
    self.targets = labels


class CIFAR10RandomSubset(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10RandomSubset, self).__init__(**kwargs)
    self.n_classes = num_classes

    labels = np.array(self.targets)
    self.data = self.data[labels < num_classes]
    labels = labels[labels < num_classes]
    labels = [int(x) for x in labels]
    self.targets = labels

    print('Number of Training Samples:', len(self.targets))

    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):

    labels = np.array(self.targets)

    print('Original Labels:', labels)

    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels

    print('Noisy Labels:', labels)

    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]
    self.targets = labels
    print(self.targets)


class CIFAR10Imbalanced(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, gamma=10, n_min=250, n_max=5000, num_classes=10, **kwargs):
    super(CIFAR10Imbalanced, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.gamma = gamma
    self.n_min = n_min
    self.n_max = n_max
    self.imbalanced_dataset()

  def imbalanced_dataset(self):

    X = np.array([[1, -self.n_max], [1, -self.n_min]])
    Y = np.array([self.n_max, self.n_min * 10 ** (self.gamma)])

    a, b = np.linalg.solve(X, Y)

    classes = list(range(1, self.num_classes + 1))

    imbal_class_counts = []
    for c in classes:
      num_c = int(np.round(a / (b + (c) ** (self.gamma))))
      print(c, num_c)
      imbal_class_counts.append(num_c)

    targets = np.array(self.targets)

    # Get class indices
    class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]

    # Get imbalanced number of instances
    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    # Set target and data to dataset
    self.targets = targets[imbal_class_indices]
    self.data = self.data[imbal_class_indices]
    print(len(self.targets))
    assert len(self.targets) == len(self.data)


if __name__ == '__main__':

  from torchvision import transforms



  transform_train = transforms.Compose([
      transforms.Pad(4, padding_mode='reflect'),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32),
      transforms.ToTensor(),
  ])


  kwargs = {'num_workers':                             1, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(
      CIFAR10Imbalanced(root='./data_imbalanced', train=True, download=True,
                          transform=transform_train, num_classes=10,
                          gamma=10),
      batch_size=200, shuffle=True)

  lst_targets = []

  for data, target in train_loader:
    lst_targets.append(target.numpy())

  targets = np.concatenate(lst_targets)
  targets.shape

  classes, class_counts = np.unique(targets, return_counts=True)
  nb_classes = len(classes)
  print(class_counts)
