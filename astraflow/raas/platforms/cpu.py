from .platform import Platform


class CpuPlatform(Platform):
    device_type: str = "cpu"
    device_control_env_var: str = ""
    communication_backend: str = "gloo"
