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

_MEAN = {"fmnist": (0.2860,), "mnist": (0.1307,), "cifar10": (0.4914, 0.4822, 0.4465),
         "femnist": (0.0381,)}
_STD = {"fmnist": (0.3530,), "mnist": (0.3081,), "cifar10": (0.2470, 0.2435, 0.2616),
        "femnist": (0.1694,)}


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


def quantity_partition(n: int, num_clients: int, sigma_q: float,
                       rng: np.random.Generator, min_size: int = 20) -> List[np.ndarray]:
    """Quantity skew with IID labels: client sizes are proportional to
    LogNormal(0, sigma_q) draws (sigma_q = 0 gives equal sizes)."""
    perm = rng.permutation(n)
    raw = rng.lognormal(0.0, sigma_q, size=num_clients)
    share = raw / raw.sum()
    sizes = np.maximum(min_size, np.floor(share * (n - min_size * num_clients)).astype(int))
    while sizes.sum() > n:
        sizes[np.argmax(sizes)] -= sizes.sum() - n
    cuts = np.cumsum(sizes)[:-1]
    return [np.sort(v) for v in np.split(perm[: sizes.sum()], cuts)]


def load_femnist(num_clients: int, seed: int, root_size: int = 100,
                 min_train: int = 100) -> FederatedData:
    """Federated EMNIST (LEAF / TFF split): one client per writer.

    ``num_clients`` writers with at least ``min_train`` training samples are
    drawn at random for each seed. Every client keeps its own writer's test
    samples as its personal test set, so per-client evaluation is on the
    writer's natural distribution. The FLTrust root set is drawn from writers
    that are *not* in the federation. Pixels are inverted so that ink is 1.
    """
    import h5py
    rng = np.random.default_rng(seed)
    ftr = h5py.File(os.path.join(DATA_ROOT, "fed_emnist_train.h5"), "r")["examples"]
    fte = h5py.File(os.path.join(DATA_ROOT, "fed_emnist_test.h5"), "r")["examples"]
    writers = sorted(ftr.keys())
    sizes = np.array([ftr[w]["label"].shape[0] for w in writers])
    elig = np.where(sizes >= min_train)[0]
    chosen = rng.choice(elig, size=num_clients, replace=False)
    others = np.setdiff1d(np.arange(len(writers)), chosen)
    xs, ys, cidx, xts, yts, tidx = [], [], [], [], [], []
    off = off_t = 0
    for j in chosen:
        w = writers[j]
        x = 1.0 - ftr[w]["pixels"][()]
        y = ftr[w]["label"][()].astype(np.int64)
        xs.append(x); ys.append(y)
        cidx.append(np.arange(off, off + len(y))); off += len(y)
        xt = 1.0 - fte[w]["pixels"][()] if w in fte else np.zeros((0, 28, 28), np.float32)
        yt = fte[w]["label"][()].astype(np.int64) if w in fte else np.zeros(0, np.int64)
        xts.append(xt); yts.append(yt)
        tidx.append(np.arange(off_t, off_t + len(yt))); off_t += len(yt)
    # root set: pooled from writers outside the federation
    rx, ry = [], []
    for j in rng.permutation(others):
        w = writers[j]
        rx.append(1.0 - ftr[w]["pixels"][()]); ry.append(ftr[w]["label"][()].astype(np.int64))
        if sum(len(v) for v in ry) >= root_size:
            break
    rx = np.concatenate(rx)[:root_size]; ry = np.concatenate(ry)[:root_size]
    root_idx = np.arange(off, off + len(ry))
    xtr = np.concatenate(xs + [rx]); ytr = np.concatenate(ys + [ry])
    xte = np.concatenate(xts); yte = np.concatenate(yts)

    def tens(x):
        t = torch.from_numpy(x).float().unsqueeze(1)
        return ((t - _MEAN["femnist"][0]) / _STD["femnist"][0]).contiguous()

    return FederatedData(tens(xtr), torch.from_numpy(ytr), tens(xte), torch.from_numpy(yte),
                         cidx, tidx, root_idx, 62)


def load_federated(dataset: str, num_clients: int, alpha: float, seed: int,
                   train_subsample: int | None = None, root_size: int = 100,
                   partition: str = "dirichlet", sigma_q: float = 1.0) -> FederatedData:
    if dataset == "femnist":
        return load_femnist(num_clients, seed, root_size)
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

    if partition == "quantity":
        local = quantity_partition(len(pool), num_clients, sigma_q, rng)
    else:
        local = dirichlet_partition(yl[pool], num_clients, alpha, rng)
    client_idx = [pool[v] for v in local]
    client_test_idx = _matched_test_split(client_idx, yl, y_test.numpy(), rng)

    return FederatedData(x_train, y_train, x_test, y_test, client_idx,
                         client_test_idx, root_idx, int(yl.max()) + 1)


def make_loader_tensors(fd: FederatedData, idx: np.ndarray) -> TensorDataset:
    return TensorDataset(fd.x_train[idx], fd.y_train[idx])


def add_trigger(x: torch.Tensor, size: int = 4, margin: int = 1) -> torch.Tensor:
    """Stamp the backdoor trigger: a size x size square at the bottom-right
    corner, set to the brightest value of each channel."""
    x = x.clone()
    hi = x.amax(dim=(0, 2, 3), keepdim=True) if x.dim() == 4 else x.max()
    H, W = x.shape[-2], x.shape[-1]
    r0, c0 = H - margin - size, W - margin - size
    x[..., r0:r0 + size, c0:c0 + size] = hi.expand_as(x[..., r0:r0 + size, c0:c0 + size])
    return x
