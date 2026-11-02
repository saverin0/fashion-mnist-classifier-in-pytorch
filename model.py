"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
# def load_fashion_mnist(n_train=10000, n_test=2000):
#     # TODO: download the four idx gz files once, parse with np.frombuffer, return float tensors in 0-1 and int64 labels.
#     pass

import os, gzip, tempfile, urllib.request
import numpy as np
import torch

BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

def _fetch(name):
    path = os.path.join(tempfile.gettempdir(), name)
    if not os.path.exists(path):
      path
      urllib.request.urlretrieve(BASE + name, path)
    return path
def _read(name, offset):
  with gzip.open(_fetch(name), "rb") as f:
    return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)

def load_fashion_mnist(n_train=10000, n_test=2000):
  train_images = _read("train-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
  train_labels = _read("train-labels-idx1-ubyte.gz", 8)
  test_images = _read("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
  test_labels = _read("t10k-labels-idx1-ubyte.gz", 8)
  return {
      "X_train" : torch.tensor(train_images[:n_train],dtype=torch.float32 ) /255,
      "y_train" : torch.tensor(train_labels[:n_train],dtype=torch.int64 ),
      "X_test" : torch.tensor(test_images[:n_test],dtype=torch.float32 ) /255,
      "y_test" : torch.tensor(test_labels[:n_test],dtype=torch.int64 )
  }

# Step 2 - FashionDataset
class FashionDataset(Dataset):
    def __init__(self, X, y, mean=0.2860, std=0.3530):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return ((self.X[i] - self.mean) / self.std, self.y[i])

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

