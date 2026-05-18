from .platform import Platform


class NPUPlatform(Platform):
    device_type: str = "npu"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    communication_backend: str = "hccl"
