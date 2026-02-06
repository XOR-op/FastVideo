# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 text-to-video pipeline.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.ltx2.ltx2_pipeline import LTX2Pipeline
from fastvideo.pipelines.stages import (
    DecodingStage,
    InputValidationStage,
    LTX2AudioDecodingStage,
    LTX2DmdDenoisingStage,
    LTX2LatentPreparationStage,
    LTX2TextEncodingStage,
)

logger = init_logger(__name__)


class LTX2DMDPipeline(LTX2Pipeline):

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=LTX2TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LTX2LatentPreparationStage(
                transformer=self.get_module("transformer"), ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2DmdDenoisingStage(
                transformer=self.get_module("transformer"), ),
        )

        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=LTX2AudioDecodingStage(
                audio_decoder=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = LTX2DMDPipeline
