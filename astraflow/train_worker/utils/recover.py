import dataclasses
import json
import os
import pickle
import shutil
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor, PreTrainedTokenizerFast

from astraflow.train_worker.api.cli_args import RecoverConfig
from astraflow.train_worker.api.engine_api import InferenceEngine, TrainEngine
from astraflow.train_worker.api.io_struct import (
    FinetuneSpec,
    SaveLoadMeta,
    StepInfo,
    WeightUpdateMeta,
)
from astraflow.train_worker.utils import logging, timeutil
from astraflow.train_worker.utils.saver import Saver

if TYPE_CHECKING:
    from astraflow.train_worker.utils.stats_logger import StatsLogger

logger = logging.getLogger("recover")


class InValidRecoverInfo(Exception):
    pass


@dataclasses.dataclass
class RecoverInfo:
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict
    buffer_info: dict[str, Any] | None = None
    rollout_dataloader_info: dict | list[dict] | None = None

    def dump(self, dump_dir: str):
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            rollout_dataloader_info = self.rollout_dataloader_info
            if rollout_dataloader_info is not None:
                rollout_dl_infos = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(rollout_dl_infos, rollout_dataloader_info)
                rollout_dataloader_info = rollout_dl_infos

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info
            rollout_dataloader_info = self.rollout_dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

        buffer_info_path = os.path.join(dump_dir, "buffer_info.pkl")
        with open(buffer_info_path, "wb") as f:
            pickle.dump(self.buffer_info, f)

        if rollout_dataloader_info is not None:
            rollout_dl_info_path = os.path.join(
                dump_dir, "rollout_dataloader_info.pkl"
            )
            with open(rollout_dl_info_path, "wb") as f:
                pickle.dump(rollout_dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            buffer_info = None
            buffer_info_path = os.path.join(load_dir, "buffer_info.pkl")
            if os.path.exists(buffer_info_path):
                with open(buffer_info_path, "rb") as f:
                    buffer_info = pickle.load(f)

            rollout_dataloader_info = None
            rollout_dl_info_path = os.path.join(
                load_dir, "rollout_dataloader_info.pkl"
            )
            if os.path.exists(rollout_dl_info_path):
                with open(rollout_dl_info_path, "rb") as f:
                    rollout_dataloader_info = pickle.load(f)
                    if isinstance(rollout_dataloader_info, list):
                        if dist.is_initialized():
                            assert dist.get_world_size() == len(
                                rollout_dataloader_info
                            ), (
                                f"Rollout dataloader info list length "
                                f"{len(rollout_dataloader_info)} does not match "
                                f"the world size {dist.get_world_size()}."
                            )
                            rollout_dataloader_info = rollout_dataloader_info[
                                dist.get_rank()
                            ]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
                buffer_info=buffer_info,
                rollout_dataloader_info=rollout_dataloader_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )
        self.last_saved_global_step: int = -1

    @staticmethod
    def recover_info_path(
        fileroot: str,
        global_step: int = -1,
    ):
        if global_step == -1:
            return os.path.join(
                Saver.get_save_root(fileroot),
                "recover_info",
            )
        else:
            return os.path.join(
                Saver.get_save_root(fileroot),
                f"recover_info_globalstep{global_step}",
            )

    @staticmethod
    def last_step_file_name(fileroot: str):
        return os.path.join(
            Saver.get_save_root(fileroot),
            "latest_step.txt",
        )

    @staticmethod
    def clean_stale_recover_data(dir_path: str, prefix):
        # 1. collect numeric step values
        steps = []
        for name in os.listdir(dir_path):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                try:
                    steps.append(int(suffix))
                except ValueError:
                    pass  # ignore malformed

        print(f"Steps: {steps}")
        # nothing to clean
        if not steps:
            return

        # 2. determine the largest step and remove others
        max_step = max(steps)
        for step in steps:
            if step != max_step:
                dir_to_remove = os.path.join(dir_path, f"{prefix}{step}")
                print("Removing:", dir_to_remove)
                shutil.rmtree(dir_to_remove, ignore_errors=True)

        print("Kept:", os.path.join(dir_path, f"{prefix}{max_step}"))

    def dump(
        self,
        engine: TrainEngine | dict[str, TrainEngine],
        step_info: StepInfo,
        saver: Saver,
        stats_logger: "StatsLogger",
        dataloader: StatefulDataLoader,
        buffer: Any | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
        rollout_dataloader: StatefulDataLoader | None = None,
    ):
        if self.config.mode == "disabled":
            return
        # currently only support recover on one engine
        if not self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        ):
            return
        if isinstance(engine, TrainEngine):
            engine = {"default": engine}

        self.last_saved_global_step = step_info.global_step

        # Release gradient memory before DCP I/O to reduce peak memory.
        # Gradients are recomputed at the start of the next train_batch().
        for engine_ in engine.values():
            if hasattr(engine_, "optimizer") and engine_.optimizer is not None:
                engine_.optimizer.zero_grad(set_to_none=True)

        for name, engine_ in engine.items():
            self._save_checkpoint(
                engine_,
                name=name,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )

        buffer_info = None
        if buffer is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            buffer_info = buffer.state_dict()

        rollout_dataloader_info = None
        if rollout_dataloader is not None:
            rollout_dataloader_info = rollout_dataloader.state_dict()

        self.last_step_info = step_info
        recover_info = RecoverInfo(
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict() if dataloader is not None else {},
            checkpoint_info=self.freq_ctl.state_dict(),
            buffer_info=buffer_info,
            rollout_dataloader_info=rollout_dataloader_info,
        )

        recover_info_path = self.recover_info_path(
            self.config.fileroot,
            global_step=self.last_saved_global_step,
        )

        recover_info.dump(recover_info_path)

        last_step_file_name = self.last_step_file_name(self.config.fileroot)
        with open(last_step_file_name, "w") as f:
            f.write(str(self.last_saved_global_step))

        # remove old checkpoints
        checkpoint_dir = Saver.get_recover_checkpoint_path(
            self.config.fileroot,
            global_step=self.last_saved_global_step,
        )
        checkpoint_parent_dir = os.path.dirname(checkpoint_dir)
        self.clean_stale_recover_data(checkpoint_parent_dir, "recover_checkpoint_globalstep")

        # remove old recover info
        recover_info_dir = self.recover_info_path(
            self.config.fileroot,
            global_step=self.last_saved_global_step,
        )
        recover_info_parent_dir = os.path.dirname(recover_info_dir)
        self.clean_stale_recover_data(recover_info_parent_dir, "recover_info_globalstep")

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine],
        saver: Saver,
        stats_logger: "StatsLogger",
        dataloader: StatefulDataLoader,
        buffer: Any | None = None,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
    ) -> RecoverInfo | None:
        if self.config.mode == "disabled":
            return

        # In "auto" mode, detect whether a recover checkpoint exists.
        # In other modes, respect the ASTRAFLOW_RECOVER_RUN env var set by launchers.
        last_step_file_name = self.last_step_file_name(self.config.fileroot)
        print(f"Last step file name: {last_step_file_name}")
        if self.config.mode == "auto":
            if not os.path.exists(last_step_file_name):
                logger.info(
                    "Recover mode=auto: no checkpoint found at %s. Starting fresh.",
                    last_step_file_name,
                )
                return
        else:
            if os.environ.get("ASTRAFLOW_RECOVER_RUN", "0") != "1":
                return

        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        if isinstance(engine, TrainEngine):
            engine = {"default": engine}

        # get last step from latest_step.txt
        with open(last_step_file_name, "r") as f:
            last_step = int(f.read())
        logger.info("Recover: last step=%d from %s", last_step, last_step_file_name)
        self.last_saved_global_step = last_step

        recover_info_path = self.recover_info_path(
            self.config.fileroot,
            global_step=last_step,
        )
        logger.info(f"Loading recover info from {recover_info_path}")


        try:
            recover_info: RecoverInfo = RecoverInfo.load(recover_info_path)
            logger.info(f"Recovering from {recover_info.last_step_info.next()}.")
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            if dataloader is not None:
                dataloader.load_state_dict(recover_info.dataloader_info)
            if buffer is not None and recover_info.buffer_info is not None:
                buffer.load_state_dict(recover_info.buffer_info)

            for name, engine_ in engine.items():
                self._load_checkpoint(engine_, name=name)
            global_step = recover_info.last_step_info.global_step

            if inference_engine is not None:
                update_engine = engine[inference_engine_update_from]
                update_engine.connect_engine(inference_engine, weight_update_meta)
                # All ranks must call update_weights — the method handles
                # rank 0 HTTP calls and barriers internally.
                update_engine.update_weights(weight_update_meta)
                update_engine.set_version(global_step + 1)
                # inference_engine.set_version is an HTTP call to the RaaS2 server,
                # so only rank 0 should call it to avoid stale-connection timeouts.
                if not dist.is_initialized() or dist.get_rank() == 0:
                    inference_engine.set_version(global_step + 1)
            return recover_info
        except (FileNotFoundError, InValidRecoverInfo):
            logger.warning(
                f"Resume info not found at {recover_info_path}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        path = Saver.get_recover_checkpoint_path(
            self.config.fileroot,
            name=name,
            global_step=self.last_saved_global_step,
        )
        print('########################################################')
        print(f"Saving recover checkpoint to {path}")
        print('########################################################')

        weight_format = "dcp"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path}")

    def _load_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        path = Saver.get_recover_checkpoint_path(
            self.config.fileroot,
            name=name,
            global_step=self.last_saved_global_step,
        )
        print('########################################################')
        print(f"Loading recover checkpoint from {path}")
        print('########################################################')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # This method is called only by launchers to check if the experiment should be a recover run
    # when "recover_mode" is auto.
    fileroot = config.fileroot

    last_step_file_name = RecoverHandler.last_step_file_name(fileroot)
    if not os.path.exists(last_step_file_name):
        return False
    with open(last_step_file_name, "r") as f:
        last_step = int(f.read())
    if last_step == 0:
        return False

    recover_info_path = RecoverHandler.recover_info_path(
        fileroot,
        global_step=last_step,
    )

    logger.info(f"Searching for recover info file in {recover_info_path}.")
    if os.path.exists(str(recover_info_path)):
        try:
            info = RecoverInfo.load(recover_info_path)
        except Exception as e:
            logger.warning(f"Failed to load recover info from {recover_info_path}: {e}")
            return False
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        save_root = Saver.get_save_root(fileroot)
        for name in os.listdir(save_root):
            if not os.path.isdir(os.path.join(save_root, name)):
                continue
            # Skip non-model directories (recover metadata lives alongside models)
            if name.startswith("recover_info"):
                continue
            path = Saver.get_recover_checkpoint_path(fileroot, name=name)
            if not os.path.exists(path):
                logger.warning(f"Recover checkpoint for model {name} does not exist.")
                return False
        return True
    logger.warning(f"Recover info not found at: {recover_info_path}")
    return False


def check_if_recover(config: RecoverConfig, run_id: int) -> bool:
    # This method is called by the launcher to check if the experiment should be a recover run
    # when "recover_mode" is not disabled.
    if config.mode == "disabled":
        return False
    elif config.mode == "auto":
        return check_if_auto_recover(config)
    elif config.mode == "fault":
        return run_id > 0
    elif config.mode == "resume":
        return True
    else:
        raise ValueError(f"Unknown recover mode: {config.mode}")
