import os

from transformers import AutoProcessor, PreTrainedTokenizerFast

from astraflow.train_worker.api.cli_args import SaverConfig
from astraflow.train_worker.api.engine_api import TrainEngine
from astraflow.train_worker.api.io_struct import FinetuneSpec, SaveLoadMeta
from astraflow.train_worker.utils import timeutil


class Saver:
    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def get_save_root(fileroot: str):
        path = os.path.join(fileroot, "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_root(
        fileroot: str,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_save_root(fileroot),
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_save_path(
        fileroot: str,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Saver.get_model_save_root(fileroot, name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_recover_checkpoint_path(
        fileroot: str,
        name: str = "default",
        global_step: int = -1,
    ):
        if global_step == -1:
            path = os.path.join(
                Saver.get_model_save_root(fileroot, name),
                "recover_checkpoint",
            )
        else:
            path = os.path.join(
                Saver.get_model_save_root(fileroot, name),
                f"recover_checkpoint_globalstep{global_step}",
            )
        os.makedirs(path, exist_ok=True)
        return path

    def state_dict(self):
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        self.freq_ctl.load_state_dict(state_dict)

    def save(
        self,
        engine: TrainEngine,
        epoch: int,
        step: int,
        global_step: int,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_spec.steps_per_epoch - 1), steps=1
        ):
            return
        path = Saver.get_model_save_path(
            self.config.fileroot,
            epoch,
            step,
            global_step,
            name,
        )
        weight_format = "hf"
        with_optim = False
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
