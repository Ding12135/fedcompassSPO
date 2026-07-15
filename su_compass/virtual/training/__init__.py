"""Pluggable client-side training extensions for virtual FL experiments."""

from .rup_adapter import RUPTrainingAdapter, RUPTrainingObservation

__all__ = ["RUPTrainingAdapter", "RUPTrainingObservation"]
