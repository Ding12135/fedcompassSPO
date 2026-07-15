from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from appfl.aggregator.fedcompass_aggregator import FedCompassAggregator


class _TinyBatchNormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.bn = torch.nn.BatchNorm1d(1)


def _aggregator(mode: str) -> FedCompassAggregator:
    config = OmegaConf.create(
        {
            "alpha": 0.8,
            "num_clients": 2,
            "staleness_fn": "constant",
            "gradient_based": True,
            "buffer_aggregation": mode,
        }
    )
    return FedCompassAggregator(_TinyBatchNormModel(), config, SimpleNamespace())


def test_weighted_single_update_does_not_replace_bn_buffer_wholesale():
    aggregator = _aggregator("weighted")
    local = aggregator.get_parameters()
    local["weight"] = torch.tensor([0.0])
    local["bn.running_mean"] = torch.tensor([10.0])
    local["bn.num_batches_tracked"] = torch.tensor(7, dtype=torch.long)

    result = aggregator.aggregate(client_id="c0", local_model=local, staleness=0)

    # alpha_t = 0.8 / 2 = 0.4
    assert torch.allclose(result["bn.running_mean"], torch.tensor([4.0]))
    assert result["bn.num_batches_tracked"].item() == 7


def test_weighted_group_buffers_use_sum_of_client_weights():
    aggregator = _aggregator("weighted")
    first = aggregator.get_parameters()
    second = aggregator.get_parameters()
    first["bn.running_mean"] = torch.tensor([2.0])
    second["bn.running_mean"] = torch.tensor([4.0])

    result = aggregator.aggregate(
        local_models={"c0": first, "c1": second},
        staleness={"c0": 0, "c1": 0},
    )

    assert torch.allclose(result["bn.running_mean"], torch.tensor([2.4]))


def test_legacy_mode_preserves_existing_single_update_behavior():
    aggregator = _aggregator("legacy")
    local = aggregator.get_parameters()
    local["bn.running_mean"] = torch.tensor([10.0])

    result = aggregator.aggregate(client_id="c0", local_model=local, staleness=0)

    assert torch.equal(result["bn.running_mean"], torch.tensor([10.0]))
