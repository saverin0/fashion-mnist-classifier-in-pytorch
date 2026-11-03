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

# Step 3 - make_loaders
def make_loaders(data, batch_size=64, val_size=2000, seed=42):
    # 1. split: last val_size training images -> validation
    X_tr = data['X_train'][:-val_size]
    X_val = data['X_train'][-val_size:]
    y_tr = data['y_train'][:-val_size]
    y_val = data['y_train'][-val_size:]

    # 2. three datasets
    ds_train = FashionDataset(X_tr, y_tr)
    ds_val = FashionDataset(X_val, y_val)
    ds_test = FashionDataset(data['X_test'],data['y_test'])

    # 3. three loaders, only train shuffled
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(ds_val, batch_size=batch_size)
    test_loader = DataLoader(ds_test, batch_size=batch_size)

    # 4. hand everything back
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "sizes": (len(ds_train), len(ds_val), len(ds_test)),
    }

# Step 4 - MLP
class MLP(nn.Module):
    def __init__(self, hidden1=300, hidden2=100, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, n_classes)
    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total

# Step 5 - train_one_epoch
def train_one_epoch(model, loader, loss_fn, optimizer):
    # TODO: model.train(); per batch: zero_grad, forward, loss, backward, step; return mean batch loss.
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad() # 1. clear old corrections
        logits = model(xb) # throw
        loss = loss_fn(logits, yb) #how far off we are
        loss.backward() #4 work out the adjusments
        optimizer.step() #adjust
        total += loss.item()
    return total / len(loader)

# Step 6 - evaluate
def evaluate(model, loader, loss_fn):
    # TODO: eval mode + no_grad; return (mean loss over examples, accuracy) as floats.
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total_loss += loss_fn(logits, yb).item() * len(yb)
            n += len(yb)
    return total_loss / n, correct / n

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

