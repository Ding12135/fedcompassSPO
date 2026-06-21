import torch

from appfl.misc.data import Dataset


def get_synthetic_mnist(
    num_clients: int,
    client_id: int,
    num_train_per_client: int = 128,
    num_test: int = 256,
    seed: int = 2025,
    **kwargs,
):
    """Small deterministic MNIST-shaped dataset for local benchmark runs."""
    _ = kwargs
    gen = torch.Generator().manual_seed(int(seed) + int(client_id))
    test_gen = torch.Generator().manual_seed(int(seed) + 100000)

    y = torch.arange(int(num_train_per_client)) % 10
    y = y[torch.randperm(len(y), generator=gen)]
    client_shift = float(client_id) / max(int(num_clients), 1)
    x = torch.randn(int(num_train_per_client), 1, 28, 28, generator=gen) * 0.25
    x = x + y.view(-1, 1, 1, 1).float() / 10.0 + client_shift

    test_y = torch.arange(int(num_test)) % 10
    test_x = torch.randn(int(num_test), 1, 28, 28, generator=test_gen) * 0.25
    test_x = test_x + test_y.view(-1, 1, 1, 1).float() / 10.0

    return Dataset(x.float(), y.long()), Dataset(test_x.float(), test_y.long())
