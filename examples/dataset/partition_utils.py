import os
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data

from appfl.misc.data import Dataset


def plot_distribution(
    num_clients: int,
    classes_samples: List[int],
    sample_matrix: np.ndarray,
    output_dirname: Optional[str],
    output_filename: Optional[str],
):
    _, ax = plt.subplots(figsize=(20, num_clients / 2 + 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ]

    for i in range(len(classes_samples)):
        ax.barh(
            y=range(num_clients),
            width=sample_matrix[i],
            left=np.sum(sample_matrix[:i], axis=0) if i > 0 else 0,
            color=colors[i % len(colors)],
        )

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])

    output_dirname = "output" if output_dirname is None else output_dirname
    output_filename = "data_distribution.pdf" if output_filename is None else output_filename
    output_filename = (
        f"{output_filename}.pdf" if not output_filename.endswith(".pdf") else output_filename
    )
    if not os.path.exists(output_dirname):
        pathlib.Path(output_dirname).mkdir(parents=True, exist_ok=True)

    unique = 1
    unique_filename = output_filename
    filename_base, ext = os.path.splitext(output_filename)
    while pathlib.Path(os.path.join(output_dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    plt.savefig(os.path.join(output_dirname, unique_filename))
    plt.close()


def _to_appfl_dataset(indices, source_dataset: data.Dataset) -> Dataset:
    data_input = []
    data_label = []
    for idx in indices:
        x, y = source_dataset[idx]
        data_input.append(x.tolist() if torch.is_tensor(x) else x)
        data_label.append(int(y))
    return Dataset(torch.FloatTensor(data_input), torch.tensor(data_label))


def iid_partition(train_dataset: data.Dataset, num_clients: int) -> List[Dataset]:
    split_indices = np.array_split(range(len(train_dataset)), num_clients)
    return [_to_appfl_dataset(split, train_dataset) for split in split_indices]


def class_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    num_classes: int = 10,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    seed: int = 42,
    **kwargs,
) -> List[Dataset]:
    np.random.seed(seed)
    cmin = {1: num_classes, 2: max(2, num_classes - 3), 3: max(2, num_classes - 4)}
    cmax = {1: num_classes, 2: max(3, num_classes - 2), 3: max(3, num_classes - 2)}
    default_cmin = max(2, num_classes // 3)
    default_cmax = max(3, num_classes // 2)

    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        label = int(label)
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    while True:
        class_partition = {}
        client_classes = {}
        for i in range(num_clients):
            cmin_i = cmin.get(num_clients, default_cmin)
            cmax_i = cmax.get(num_clients, default_cmax)
            cnum = np.random.randint(cmin_i, min(cmax_i, num_classes) + 1)
            classes = np.random.permutation(range(num_classes))[:cnum]
            client_classes[i] = classes
            for cls in classes:
                class_partition[cls] = class_partition.get(cls, 0) + 1
        if len(class_partition) == num_classes:
            break

    partition_endpoints = {}
    for label in labels:
        total_size = len(label_indices[label])
        partitions = class_partition[label]
        partition_lengths = np.abs(np.random.normal(10, 3, size=partitions))
        partition_lengths = partition_lengths / np.sum(partition_lengths) * total_size
        endpoints = np.cumsum(partition_lengths).astype(np.int32)
        endpoints[-1] = total_size
        partition_endpoints[label] = endpoints

    partition_pointer = {label: 0 for label in labels}
    client_datasets = []
    client_dataset_info = {}
    for i in range(num_clients):
        client_dataset_info[i] = {}
        sample_indices = []
        for cls in client_classes[i]:
            start_idx = (
                0
                if partition_pointer[cls] == 0
                else partition_endpoints[cls][partition_pointer[cls] - 1]
            )
            end_idx = partition_endpoints[cls][partition_pointer[cls]]
            sample_indices.extend(label_indices[cls][start_idx:end_idx])
            partition_pointer[cls] += 1
            client_dataset_info[i][cls] = end_idx - start_idx
        client_datasets.append(sample_indices)

    if visualization:
        classes_samples = [len(label_indices[label]) for label in labels]
        sample_matrix = np.zeros((len(classes_samples), num_clients))
        for i in range(num_clients):
            for cls in client_dataset_info[i]:
                sample_matrix[cls][i] = client_dataset_info[i][cls]
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    return [_to_appfl_dataset(indices, train_dataset) for indices in client_datasets]


def dirichlet_noniid_partition(
    train_dataset: data.Dataset,
    num_clients: int,
    visualization: bool = False,
    output_dirname: Optional[str] = None,
    output_filename: Optional[str] = None,
    alpha1: float = 8.0,
    alpha2: float = 0.5,
    dirichlet_mode: str = "legacy_two_level",
    seed: int = 42,
    **kwargs,
) -> List[Dataset]:
    np.random.seed(seed)
    labels = []
    label_indices = {}
    for idx, (_, label) in enumerate(train_dataset):
        label = int(label)
        if label not in label_indices:
            label_indices[label] = []
            labels.append(label)
        label_indices[label].append(idx)
    labels.sort()

    for label in labels:
        np.random.shuffle(label_indices[label])
    classes_samples = [len(label_indices[label]) for label in labels]

    if dirichlet_mode == "standard":
        if alpha2 <= 0:
            raise ValueError("alpha2 must be positive for standard Dirichlet partitioning")
        # Conventional label-wise Dirichlet partition: for every class, draw
        # client proportions from Dirichlet(alpha).  Thus alpha2=0.5 really
        # means concentration 0.5 per client, matching the common FL protocol.
        sample_matrix = np.zeros((len(labels), num_clients), dtype=np.int64)
        for row, label in enumerate(labels):
            proportions = np.random.dirichlet([alpha2] * num_clients)
            counts = np.floor(proportions * len(label_indices[label])).astype(np.int64)
            remainder = len(label_indices[label]) - int(counts.sum())
            if remainder:
                fractional = proportions * len(label_indices[label]) - counts
                for client_idx in np.argsort(-fractional)[:remainder]:
                    counts[client_idx] += 1
            sample_matrix[row] = counts
    elif dirichlet_mode != "legacy_two_level":
        raise ValueError(
            "dirichlet_mode must be 'legacy_two_level' or 'standard'"
        )

    if dirichlet_mode == "legacy_two_level":
        p1 = [1 / num_clients for _ in range(num_clients)]
        p2 = [len(label_indices[label]) for label in labels]
        p2 = [p / sum(p2) for p in p2]
        q1 = [alpha1 * i for i in p1]
        q2 = [alpha2 * i for i in p2]

        weights = np.random.dirichlet(q1)
        individuals = np.random.dirichlet(q2, num_clients)
        normalized_portions = np.zeros(individuals.shape)
        for i in range(num_clients):
            for j in range(len(classes_samples)):
                normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(
                    weights, individuals.transpose()[j]
                )

        sample_matrix = np.multiply(
            np.array([classes_samples] * num_clients), normalized_portions
        ).transpose()

        for i in range(len(classes_samples)):
            total = 0
            for j in range(num_clients - 1):
                sample_matrix[i][j] = int(sample_matrix[i][j])
                total += sample_matrix[i][j]
            sample_matrix[i][num_clients - 1] = classes_samples[i] - total

    if visualization:
        plot_distribution(
            num_clients, classes_samples, sample_matrix, output_dirname, output_filename
        )

    num_elements = np.array(sample_matrix.transpose(), dtype=np.int32)
    sum_elements = np.cumsum(num_elements, axis=0)

    train_datasets = []
    for i in range(num_clients):
        sample_indices = []
        for j, label in enumerate(labels):
            start = 0 if i == 0 else sum_elements[i - 1][j]
            end = sum_elements[i][j]
            sample_indices.extend(label_indices[label][start:end])
        train_datasets.append(_to_appfl_dataset(sample_indices, train_dataset))
    return train_datasets


def partition_dataset(
    train_dataset: data.Dataset,
    num_clients: int,
    partition_strategy: str,
    num_classes: int = 10,
    **kwargs,
) -> List[Dataset]:
    strategy = partition_strategy.lower()
    if strategy == "iid":
        return iid_partition(train_dataset, num_clients)
    if strategy in {"class_noniid", "noniid"}:
        return class_noniid_partition(
            train_dataset, num_clients, num_classes=num_classes, **kwargs
        )
    if strategy in {"dirichlet_noniid", "dirichlet_nomiid"}:
        return dirichlet_noniid_partition(train_dataset, num_clients, **kwargs)
    raise ValueError(f"Invalid partition strategy: {partition_strategy}")
