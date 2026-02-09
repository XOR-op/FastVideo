# SPDX-License-Identifier: Apache-2.0
import sys
from copy import deepcopy
from typing import cast

import torch

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.ltx2.ltx2_dmd_pipeline import LTX2DMDPipeline
from fastvideo.training.distillation_pipeline import DistillationPipeline
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.utils import is_vsa_available

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
        super().__init__(*args, **kwargs)
        self.modules["scheduler"] = torch.nn.Linear(
            1, 1)  # dummy scheduler module to satisfy required modules

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
