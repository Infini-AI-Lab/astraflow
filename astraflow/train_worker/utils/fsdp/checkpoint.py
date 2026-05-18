"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class DCPState(Stateful):
    """Wrapper for checkpointing the State using DCP.

    This class implements the Stateful protocol, so DCP will automatically call
    state_dict/load_state_dict as needed in the dcp.save/load APIs.

    It handles calling distributed state dict methods on the model and optimizer.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dict for model and optimizer using DCP utilities.
        This automatically manages FSDP FQN's and
        sets default state dict type to FSDP.SHARDED_STATE_DICT
        """
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
        else:
            state_dict = {"model": get_model_state_dict(self.model)}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state dicts onto model and optimizer.
        """
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )


class LoRADCPState(Stateful):
    """DCP state wrapper that saves/loads only LoRA adapter weights.

    Base model weights are frozen and loaded from the original model path
    during engine init, so they don't need to be checkpointed.

    Optimizer state is naturally LoRA-only (frozen params have no grad state).
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ):
        self.model = model
        self.optimizer = optimizer

    @staticmethod
    def _filter_lora_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep only LoRA adapter keys from a model state dict."""
        return {k: v for k, v in state_dict.items() if "lora_" in k}

    def state_dict(self) -> dict[str, Any]:
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            state_dict = {
                "model": self._filter_lora_keys(model_state_dict),
                "optim": optimizer_state_dict,
            }
        else:
            model_state_dict = get_model_state_dict(self.model)
            state_dict = {"model": self._filter_lora_keys(model_state_dict)}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(strict=False),
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
                options=StateDictOptions(strict=False),
            )
