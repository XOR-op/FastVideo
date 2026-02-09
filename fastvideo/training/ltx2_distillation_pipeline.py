# SPDX-License-Identifier: Apache-2.0
import os
import sys
from copy import deepcopy
from typing import cast

import torch

from fastvideo.dataset.parquet_dataset_map_style import (
    build_parquet_map_style_dataloader, )
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2v
from fastvideo.distributed.parallel_state import (
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.pipelines.basic.ltx2.ltx2_dmd_pipeline import LTX2DMDPipeline
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.trackers import (
    DummyTracker,
    Trackers,
    initialize_trackers,
)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import EMA_FSDP, get_scheduler
from fastvideo.utils import is_vsa_available, set_random_seed

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class LTX2DistillationPipeline(DistillationPipeline):
    """DMD distillation pipeline for LTX-2 using precomputed LTX-2 data."""

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    with_audio: bool = True

    def __init__(self, *args, **kwargs):
        self.real_score_transformer: torch.nn.Module = None  # type: ignore[assignment]
        self.fake_score_transformer: torch.nn.Module = None  # type: ignore[assignment]
        super().__init__(*args, **kwargs)

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        self._training_initialize_training_pipeline(training_args)
        self._distillation_initialize_training_pipeline(training_args)

    def _training_initialize_training_pipeline(self,
                                               training_args: TrainingArgs):
        logger.info("Initializing LTX-2 base pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank
        self.transformer = self.get_module("transformer")
        self.transformer_2 = self.get_module("transformer_2", None)
        self.text_encoder = self.get_module("text_encoder")
        self.text_encoder.eval()
        self.text_encoder.to(self.device)
        self.seed = training_args.seed

        assert self.seed is not None, "seed must be set"
        set_random_seed(self.seed)
        self.transformer.train()

        if training_args.enable_gradient_checkpointing_type is not None:
            from fastvideo.training.activation_checkpoint import (
                apply_activation_checkpointing, )

            self.transformer = apply_activation_checkpointing(
                self.transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type,
            )

        self.set_trainable()
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters()))
        betas_str = training_args.betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=training_args.learning_rate,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.init_steps = 0
        logger.info("optimizer: %s", self.optimizer)

        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        self.with_audio = True
        self.train_dataset_schema = pyarrow_schema_t2v
        self.train_dataset, self.train_dataloader = (
            build_parquet_map_style_dataloader(
                training_args.data_path,
                training_args.train_batch_size,
                parquet_schema=self.train_dataset_schema,
                num_data_workers=training_args.dataloader_num_workers,
                cfg_rate=training_args.training_cfg_rate,
                drop_last=True,
                text_padding_length=training_args.pipeline_config.
                text_encoder_configs[0].arch_config.
                text_len,  # type: ignore[attr-defined]
                seed=self.seed,
            ))

        self.num_update_steps_per_epoch = max(
            1,
            len(self.train_dataloader) //
            training_args.gradient_accumulation_steps,
        )
        self.num_train_epochs = max(
            1, training_args.max_train_steps // self.num_update_steps_per_epoch)
        self.current_epoch = 0

        trackers = list(training_args.trackers)
        if not trackers and training_args.tracker_project_name:
            trackers.append(Trackers.WANDB.value)
        if self.global_rank != 0:
            trackers = []

        tracker_log_dir = training_args.output_dir or os.getcwd()
        if trackers:
            tracker_log_dir = os.path.join(tracker_log_dir, "tracker")

        tracker_config = training_args.__dict__ if trackers else None
        tracker_run_name = training_args.wandb_run_name or None
        project = training_args.tracker_project_name or "fastvideo"
        self.tracker = (initialize_trackers(
            trackers,
            experiment_name=project,
            config=tracker_config,
            log_dir=tracker_log_dir,
            run_name=tracker_run_name,
        ) if trackers else DummyTracker())

    def _distillation_initialize_training_pipeline(self,
                                                   training_args: TrainingArgs):
        self.vae = self.get_module("vae")
        self.vae.requires_grad_(False)

        self.timestep_shift = self.training_args.pipeline_config.flow_shift
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=self.timestep_shift)

        if self.training_args.boundary_ratio is not None:
            self.boundary_timestep = (self.training_args.boundary_ratio *
                                      self.noise_scheduler.num_train_timesteps)
        else:
            self.boundary_timestep = None

        # make sure the real score transformer is not trainable
        self.real_score_transformer.requires_grad_(False)
        self.real_score_transformer.eval()

        if training_args.enable_gradient_checkpointing_type is not None:
            self.fake_score_transformer = apply_activation_checkpointing(
                self.fake_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type,
            )

            self.real_score_transformer = apply_activation_checkpointing(
                self.real_score_transformer,
                checkpointing_type=training_args.
                enable_gradient_checkpointing_type,
            )

        # Initialize optimizers
        fake_score_params = list(
            filter(
                lambda p: p.requires_grad,
                self.fake_score_transformer.parameters(),
            ))

        # Use separate learning rate for fake_score_transformer if specified
        fake_score_lr = training_args.fake_score_learning_rate
        if fake_score_lr == 0.0:
            fake_score_lr = training_args.learning_rate

        betas_str = training_args.fake_score_betas
        betas = tuple(float(x.strip()) for x in betas_str.split(","))

        self.fake_score_optimizer = torch.optim.AdamW(
            fake_score_params,
            lr=fake_score_lr,
            betas=betas,
            weight_decay=training_args.weight_decay,
            eps=1e-8,
        )

        self.fake_score_lr_scheduler = get_scheduler(
            training_args.fake_score_lr_scheduler,
            optimizer=self.fake_score_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=self.init_steps - 1,
        )

        logger.info(
            "Distillation optimizers initialized: generator and fake_score")

        self.generator_update_interval = (
            self.training_args.generator_update_interval)
        logger.info(
            "Distillation pipeline initialized with generator_update_interval=%s",
            self.generator_update_interval,
        )

        self.denoising_step_list = torch.tensor(
            self.training_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device(),
        )

        if (training_args.warp_denoising_step
            ):  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((
                self.noise_scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32),
            )).cuda()
            self.denoising_step_list = timesteps[1000 -
                                                 self.denoising_step_list]
            logger.info("Warping denoising_step_list")

        self.denoising_step_list = self.denoising_step_list.to(
            get_local_torch_device())
        logger.info(
            "Distillation generator model to %s denoising steps: %s",
            len(self.denoising_step_list),
            self.denoising_step_list,
        )
        self.num_train_timestep = self.noise_scheduler.num_train_timesteps

        self.min_timestep = int(self.training_args.min_timestep_ratio *
                                self.num_train_timestep)
        self.max_timestep = int(self.training_args.max_timestep_ratio *
                                self.num_train_timestep)

        self.real_score_guidance_scale = (
            self.training_args.real_score_guidance_scale)

        self.generator_ema: EMA_FSDP | None = None
        if (self.training_args.ema_decay
                is not None) and (self.training_args.ema_decay > 0.0):
            self.generator_ema = EMA_FSDP(self.transformer,
                                          decay=self.training_args.ema_decay)
            logger.info(
                "Initialized generator EMA with decay=%s",
                self.training_args.ema_decay,
            )
        else:
            logger.info("Generator EMA disabled (ema_decay <= 0.0)")

    def _normalize_dit_input(self,
                             training_batch: TrainingBatch) -> TrainingBatch:
        # just skip
        return training_batch

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
        #     shift=fastvideo_args.pipeline_config.flow_shift)
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ):
        # bypass DistillationPipeline's load_modules()
        modules = TrainingPipeline.load_modules(self, fastvideo_args,
                                                loaded_modules)
        training_args = cast(TrainingArgs, fastvideo_args)

        if training_args.real_score_model_path:
            logger.info(
                "Loading real score transformer from: %s",
                training_args.real_score_model_path,
            )
            # TODO(will): can use deepcopy instead if the model is the same
            self.real_score_transformer = self.load_module_from_path(
                training_args.real_score_model_path,
                "transformer",
                training_args,
            )
            modules["real_score_transformer"] = self.real_score_transformer
            self.real_score_transformer_2 = None
        else:
            raise ValueError(
                "real_score_model_path is required for DMD distillation pipeline"
            )

        if training_args.fake_score_model_path:
            logger.info(
                "Loading fake score transformer from: %s",
                training_args.fake_score_model_path,
            )
            self.fake_score_transformer = self.load_module_from_path(
                training_args.fake_score_model_path,
                "transformer",
                training_args,
            )
            modules["fake_score_transformer"] = self.fake_score_transformer
            self.fake_score_transformer_2 = None
        else:
            raise ValueError(
                "fake_score_model_path is required for DMD distillation pipeline"
            )

        return modules

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        validation_pipeline = LTX2DMDPipeline.from_pretrained(
            training_args.model_path,
            args=args_copy,  # type: ignore
            inference_mode=True,
            loaded_modules={"transformer": self.get_module("transformer")},
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            pin_cpu_memory=training_args.pin_cpu_memory,
            dit_cpu_offload=False,
            dit_layerwise_offload=True,
        )

        self.validation_pipeline = validation_pipeline


def main(args) -> None:
    logger.info("Starting LTX-2 distillation pipeline...")
    pipeline = LTX2DistillationPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("LTX-2 distillation pipeline completed")


if __name__ == "__main__":
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    args.dit_layerwise_offload = False
    main(args)
