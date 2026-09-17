"""Dataset loading and non-IID partitioning for the federated simulator."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_MEAN = {"fmnist": (0.2860,), "mnist": (0.1307,), "cifar10": (0.4914, 0.4822, 0.4465)}
_STD = {"fmnist": (0.3530,), "mnist": (0.3081,), "cifar10": (0.2470, 0.2435, 0.2616)}


@dataclass
class FederatedData:
    """Tensorised federation: everything lives in RAM as normalised tensors."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    client_idx: List[np.ndarray]          # train indices held by each client
    client_test_idx: List[np.ndarray]     # test indices matching each client's label mix
    root_idx: np.ndarray                  # small server-side clean set (FLTrust only)
    num_classes: int

    @property
    def num_clients(self) -> int:
        return len(self.client_idx)

    def client_sizes(self) -> np.ndarray:
        return np.array([len(ix) for ix in self.client_idx], dtype=np.float64)


def _to_tensors(ds, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if name in ("fmnist", "mnist"):
        x = ds.data.float().div_(255.0).unsqueeze(1)
        y = ds.targets.clone().long()
    else:
        x = torch.from_numpy(ds.data).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets, dtype=torch.long)
    mean = torch.tensor(_MEAN[name]).view(1, -1, 1, 1)
    std = torch.tensor(_STD[name]).view(1, -1, 1, 1)
    x = (x - mean) / std
    return x.contiguous(), y


def _load_raw(name: str):
    tf = transforms.ToTensor()
    if name == "fmnist":
        tr = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tf)
    elif name == "mnist":
        tr = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
    elif name == "cifar10":
        tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
        te = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
    else:
        raise ValueError(name)
    return tr, te


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        rng: np.random.Generator, min_size: int = 20) -> List[np.ndarray]:
    """Label-skewed partition: client k receives a Dir(alpha) share of every class.

    This is the standard non-IID protocol of Hsu et al.; alpha -> 0 gives a
    pathological single-class-per-client split, alpha -> inf gives IID.
    """
    n_classes = int(labels.max()) + 1
    while True:
        idx_per_client: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[k].extend(part.tolist())
        sizes = [len(v) for v in idx_per_client]
        if min(sizes) >= min_size:
            break
    return [np.array(sorted(v)) for v in idx_per_client]


def _matched_test_split(train_idx: List[np.ndarray], y_train: np.ndarray,
                        y_test: np.ndarray, rng: np.random.Generator,
                        per_client: int = 300) -> List[np.ndarray]:
    """Give every client a personal test set whose label mix matches its own.

    Per-client accuracy on these sets is what the fairness index is computed on:
    it measures whether the global model serves each client's own distribution.
    """
    n_classes = int(y_train.max()) + 1
    by_class = [np.where(y_test == c)[0] for c in range(n_classes)]
    out = []
    for ix in train_idx:
        counts = np.bincount(y_train[ix], minlength=n_classes).astype(np.float64)
        p = counts / counts.sum()
        draw = rng.multinomial(per_client, p)
        sel = []
        for c in range(n_classes):
            if draw[c] == 0 or len(by_class[c]) == 0:
                continue
            sel.append(rng.choice(by_class[c], size=min(draw[c], len(by_class[c])),
                                  replace=draw[c] > len(by_class[c])))
        out.append(np.concatenate(sel) if sel else rng.choice(len(y_test), 10))
    return out


def load_federated(dataset: str, num_clients: int, alpha: float, seed: int,
                   train_subsample: int | None = None, root_size: int = 100) -> FederatedData:
    rng = np.random.default_rng(seed)
    tr, te = _load_raw(dataset)
    x_train, y_train = _to_tensors(tr, dataset)
    x_test, y_test = _to_tensors(te, dataset)

    yl = y_train.numpy()
    all_idx = np.arange(len(yl))

    # Carve out a small clean server-side root set before partitioning so that no
    # client and the server ever share a sample.
    rng.shuffle(all_idx)
    root_idx = all_idx[:root_size]
    pool = all_idx[root_size:]
    if train_subsample is not None and train_subsample < len(pool):
        pool = pool[:train_subsample]
    pool = np.sort(pool)

    local = dirichlet_partition(yl[pool], num_clients, alpha, rng)
    client_idx = [pool[v] for v in local]
    client_test_idx = _matched_test_split(client_idx, yl, y_test.numpy(), rng)

    return FederatedData(x_train, y_train, x_test, y_test, client_idx,
                         client_test_idx, root_idx, int(yl.max()) + 1)


def make_loader_tensors(fd: FederatedData, idx: np.ndarray) -> TensorDataset:
    return TensorDataset(fd.x_train[idx], fd.y_train[idx])
