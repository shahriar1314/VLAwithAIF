from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@dataclass
class CalvinBatch:
    image: torch.Tensor              # [B, 3, H, W]
    proprio: torch.Tensor            # [B, P], in your setup P = 15
    action: Optional[torch.Tensor]   # [B, 7], optional for live env obs
    instruction: List[str]           # list of language instructions


class CalvinPreprocessor:
    """
    Converts CALVIN observations or dataset timesteps into clean tensors.

    Your current CALVIN env observation looks like:
        obs["rgb_obs"]["rgb_static"]
        obs["rgb_obs"]["rgb_gripper"]
        obs["robot_obs"]              # shape: [15]

    Dataset .npz files usually contain:
        data["rgb_static"]
        data["rgb_gripper"]
        data["robot_obs"]
        data["rel_actions"]           # shape: [7]
    """

    def __init__(
        self,
        image_size: int = 224,
        camera_key: str = "rgb_static",
    ):
        self.image_size = image_size
        self.camera_key = camera_key

        # Keep image in [0, 1]. Do not normalize yet.
        # OpenVLA-OFT may use its own processor later.
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def image_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to torch image tensor [3, H, W].
        Handles uint8, float, HWC, and CHW images.
        """
        img_np = np.asarray(img_np)

        if img_np.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {img_np.shape}")

        # Convert CHW to HWC if needed
        if img_np.shape[0] in [1, 3, 4] and img_np.shape[-1] not in [1, 3, 4]:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Remove alpha channel if present
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # Convert grayscale to RGB if needed
        if img_np.shape[-1] == 1:
            img_np = np.repeat(img_np, 3, axis=-1)

        # Convert to uint8
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.float32)

            # If image is in [0, 1], convert to [0, 255]
            if img_np.max() <= 1.0:
                img_np = img_np * 255.0

            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_np)
        return self.image_tf(img)

    def get_image_from_obs(self, obs: Dict[str, Any], camera_key: Optional[str] = None) -> np.ndarray:
        """
        Extract image from live CALVIN env observation.
        """
        key = camera_key or self.camera_key

        # Your current live CALVIN format
        if "rgb_obs" in obs and key in obs["rgb_obs"]:
            return obs["rgb_obs"][key]

        # Fallback for flat dict format
        if key in obs:
            return obs[key]

        raise KeyError(
            f"Could not find image '{key}'. Available top-level keys: {list(obs.keys())}"
        )

    def obs_to_model_input(
        self,
        obs: Dict[str, Any],
        instruction: str,
        action: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Convert one live CALVIN environment observation into model input.
        """
        image_np = self.get_image_from_obs(obs, self.camera_key)
        image = self.image_to_tensor(image_np)

        if "robot_obs" not in obs:
            raise KeyError(f"Missing robot_obs. Available keys: {list(obs.keys())}")

        proprio = torch.tensor(obs["robot_obs"], dtype=torch.float32)

        model_input = {
            "image": image,                       # [3, 224, 224]
            "proprio": proprio,                   # [15]
            "instruction": instruction,            # string
        }

        if action is not None:
            model_input["action"] = torch.tensor(action, dtype=torch.float32)

        return model_input

    def _select_timestep(self, value: np.ndarray, t: int):
        """
        CALVIN debug .npz files can be single-step files:
            rgb_static:  [H, W, 3]
            robot_obs:   [15]
            rel_actions: [7]

        Some full datasets/loaders may provide sequence arrays:
            rgb_static:  [T, H, W, 3]
            robot_obs:   [T, 15]
            rel_actions: [T, 7]

        This helper supports both.
        """
        value = np.asarray(value)

        # Image already single-step: [H, W, C]
        if value.ndim == 3 and value.shape[-1] in [1, 3, 4, 6]:
            return value

        # Vector already single-step: [D]
        if value.ndim == 1:
            return value

        # Sequence: [T, ...]
        return value[t]

    def dataset_timestep_to_model_input(
        self,
        data: Dict[str, Any],
        t: int,
        instruction: str,
        action_key: str = "rel_actions",
    ) -> Dict[str, Any]:
        """
        Convert one timestep from a CALVIN .npz file into model input.

        Works for both:
        1. single-step .npz files from calvin_debug_dataset
        2. sequence-style arrays if available later
        """
        if self.camera_key not in data:
            raise KeyError(
                f"Missing {self.camera_key}. Available dataset keys: {list(data.keys())}"
            )

        image_np = self._select_timestep(data[self.camera_key], t)
        robot_obs_np = self._select_timestep(data["robot_obs"], t)

        image = self.image_to_tensor(image_np)
        proprio = torch.tensor(robot_obs_np, dtype=torch.float32)

        model_input = {
            "image": image,
            "proprio": proprio,
            "instruction": instruction,
        }

        if action_key in data:
            action_np = self._select_timestep(data[action_key], t)
            model_input["action"] = torch.tensor(action_np, dtype=torch.float32)

        return model_input

    def collate(self, items: List[Dict[str, Any]]) -> CalvinBatch:
        """
        Convert list of single-step inputs into a batch.
        """
        images = torch.stack([item["image"] for item in items], dim=0)
        proprios = torch.stack([item["proprio"] for item in items], dim=0)
        instructions = [item["instruction"] for item in items]

        actions = None
        if all("action" in item for item in items):
            actions = torch.stack([item["action"] for item in items], dim=0)

        return CalvinBatch(
            image=images,
            proprio=proprios,
            action=actions,
            instruction=instructions,
        )


def load_npz_episode(path: str) -> Dict[str, np.ndarray]:
    """
    Load one CALVIN episode_*.npz file.
    """
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}