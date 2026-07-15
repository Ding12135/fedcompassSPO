import importlib.util
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples/dataset/partition_utils.py"
SPEC = importlib.util.spec_from_file_location("partition_utils_under_test", MODULE_PATH)
PARTITION_UTILS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARTITION_UTILS)


def _balanced_dataset(samples_per_class: int = 40):
    labels = torch.arange(4).repeat_interleave(samples_per_class)
    features = torch.arange(len(labels), dtype=torch.float32).reshape(-1, 1)
    return TensorDataset(features, labels)


def test_standard_dirichlet_is_deterministic_and_preserves_all_samples():
    dataset = _balanced_dataset()
    kwargs = dict(
        num_clients=4,
        alpha2=0.5,
        dirichlet_mode="standard",
        seed=42,
    )
    first = PARTITION_UTILS.dirichlet_noniid_partition(dataset, **kwargs)
    second = PARTITION_UTILS.dirichlet_noniid_partition(dataset, **kwargs)

    assert sum(len(part) for part in first) == len(dataset)
    assert [part.y.tolist() for part in first] == [part.y.tolist() for part in second]


def test_unknown_dirichlet_mode_is_rejected():
    dataset = _balanced_dataset()
    try:
        PARTITION_UTILS.dirichlet_noniid_partition(
            dataset, num_clients=4, dirichlet_mode="unknown"
        )
    except ValueError as error:
        assert "dirichlet_mode" in str(error)
    else:
        raise AssertionError("unknown mode must raise ValueError")
