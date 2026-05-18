from .platform import Platform


class CudaPlatform(Platform):
    device_type: str = "cuda"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"
