import torch


class Platform:
    """Hardware platform abstraction providing device type, communication backend,
    and transparent access to torch.<device_type> APIs via __getattr__.
    """

    # Torch device module name, e.g. "cuda", "npu", "cpu"
    device_type: str

    # Environment variable controlling device visibility,
    # e.g. "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"
    device_control_env_var: str

    # Distributed communication backend, e.g. "nccl", "hccl", "gloo"
    communication_backend: str

    def __getattr__(self, key: str):
        """Delegate to torch.<device_type> for device-specific APIs."""
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
