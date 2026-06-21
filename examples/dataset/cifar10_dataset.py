import os

import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from appfl.misc.data import Dataset
from partition_utils import partition_dataset


def _raw_data_dir() -> str:
    return os.path.join(os.getcwd(), "datasets", "RawData")


def _build_test_dataset(test_data_raw: data.Dataset) -> Dataset:
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        x, y = test_data_raw[idx]
        test_data_input.append(x.tolist())
        test_data_label.append(int(y))
    return Dataset(
        __import__("torch").FloatTensor(test_data_input),
        __import__("torch").tensor(test_data_label),
    )


def get_cifar10(
    num_clients: int,
    client_id: int,
    partition_strategy: str = "dirichlet_noniid",
    seed: int = 42,
    **kwargs,
):
    """
    Return the CIFAR-10 dataset for a given client.

    partition_strategy:
      - iid
      - class_noniid / noniid
      - dirichlet_noniid
    """
    data_dir = _raw_data_dir()
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_data_raw = torchvision.datasets.CIFAR10(
        data_dir, download=True, train=False, transform=test_transform
    )
    train_data_raw = torchvision.datasets.CIFAR10(
        data_dir, download=False, train=True, transform=train_transform
    )

    test_dataset = _build_test_dataset(test_data_raw)
    train_datasets = partition_dataset(
        train_data_raw,
        num_clients=num_clients,
        partition_strategy=partition_strategy,
        num_classes=10,
        seed=seed,
        **kwargs,
    )
    return train_datasets[client_id], test_dataset
