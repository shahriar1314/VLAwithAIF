from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from aif_vla.libero_adapter import LiberoModelInput


@dataclass
class OpenVLAOFTAdapterConfig:
    pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    unnorm_key: str = "libero_spatial_no_noops"

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False

    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True

    load_in_8bit: bool = False
    load_in_4bit: bool = True


class OpenVLAOFTAdapter:
    """
    Thin wrapper around OpenVLA-OFT official utilities.

    Input:
        LiberoModelInput

    Output:
        action_chunk: np.ndarray with shape [8, 7]
    """

    def __init__(self, config: Optional[OpenVLAOFTAdapterConfig] = None):
        self.config = config or OpenVLAOFTAdapterConfig()

        # Lazy imports so this file can exist even when OpenVLA-OFT is not installed.
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

        self._get_vla_action = get_vla_action

        self.cfg = GenerateConfig(
            pretrained_checkpoint=self.config.pretrained_checkpoint,
            use_l1_regression=self.config.use_l1_regression,
            use_diffusion=self.config.use_diffusion,
            use_film=self.config.use_film,
            num_images_in_input=self.config.num_images_in_input,
            use_proprio=self.config.use_proprio,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            center_crop=self.config.center_crop,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key=self.config.unnorm_key,
        )

        print("Loading OpenVLA-OFT model...")
        self.vla = get_vla(self.cfg)
        print("OpenVLA-OFT model loaded.")

        print("Loading OpenVLA-OFT processor...")
        self.processor = get_processor(self.cfg)
        print("Processor loaded.")

        print("Loading OpenVLA-OFT action head...")
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        print("Action head loaded.")

        print("Loading OpenVLA-OFT proprio projector...")
        self.proprio_projector = get_proprio_projector(
            self.cfg,
            llm_dim=self.vla.llm_dim,
            proprio_dim=PROPRIO_DIM,
        )
        print("Proprio projector loaded.")

    def predict_action_chunk(self, model_input: LiberoModelInput) -> np.ndarray:
        observation = model_input.to_openvla_observation()
        instruction = model_input.instruction

        with torch.inference_mode():
            action_chunk = self._get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                observation,
                instruction,
                self.action_head,
                self.proprio_projector,
            )

        action_chunk = np.asarray(action_chunk, dtype=np.float32)

        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk [T, D], got {action_chunk.shape}")

        if action_chunk.shape[-1] != 7:
            raise ValueError(f"Expected 7D actions, got {action_chunk.shape}")

        return action_chunk
