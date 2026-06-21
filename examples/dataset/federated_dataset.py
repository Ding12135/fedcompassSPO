"""
Unified dataset entry points for benchmark experiments.

Supported datasets:
  - mnist
  - cifar10
  - synthetic
"""

from mnist_dataset import get_mnist
from cifar10_dataset import get_cifar10
from synthetic_mnist_dataset import get_synthetic_mnist


def get_federated_dataset(
    dataset: str,
    num_clients: int,
    client_id: int,
    partition_strategy: str = "dirichlet_noniid",
    seed: int = 42,
    **kwargs,
):
    name = dataset.lower()
    if name == "mnist":
        return get_mnist(
            num_clients=num_clients,
            client_id=client_id,
            partition_strategy=partition_strategy,
            seed=seed,
            **kwargs,
        )
    if name in {"cifar10", "cifar-10"}:
        return get_cifar10(
            num_clients=num_clients,
            client_id=client_id,
            partition_strategy=partition_strategy,
            seed=seed,
            **kwargs,
        )
    if name in {"synthetic", "synthetic_mnist"}:
        return get_synthetic_mnist(
            num_clients=num_clients,
            client_id=client_id,
            seed=seed,
            **kwargs,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
