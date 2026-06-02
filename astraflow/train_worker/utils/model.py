import torch


VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_5",
    "gemma3",
]
# Registry of vision models verified to work with this framework.
# Different vision models vary in image processing, special tokens and keys, etc.
# We add models to this registry as we test them.
# If you want to add a new vision model, please verify it works end-to-end first.


def is_valid_vision_model(model_type: str) -> bool:
    return model_type in VALID_VISION_MODELS


def is_qwen2_vl_model(model_type: str) -> bool:
    return model_type in ["qwen2_vl", "qwen2_5_vl"]


def is_qwen3_vl_model(model_type: str) -> bool:
    return model_type in ["qwen3_vl"]


def is_qwen3_5_model(model_type: str) -> bool:
    return model_type in ["qwen3_5"]


def is_qwen_vl_model(model_type: str) -> bool:
    return is_qwen2_vl_model(model_type) or is_qwen3_vl_model(model_type)


def is_gemma3_model(model_type: str) -> bool:
    return model_type in ["gemma3"]


VALID_MOE_MODELS = [
    "qwen3_moe",
]
# Registry of MoE models verified to work with this framework.


def is_moe_model(model_type: str) -> bool:
    return model_type in VALID_MOE_MODELS


def is_qwen3_moe_model(model_type: str) -> bool:
    return model_type in ["qwen3_moe"]


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


