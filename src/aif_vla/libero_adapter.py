from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class LiberoModelInput:
    full_image: np.ndarray
    wrist_image: np.ndarray
    proprio: np.ndarray
    instruction: str

    def to_openvla_observation(self) -> Dict[str, Any]:
        """
        Format expected by OpenVLA-OFT get_vla_action().
        """
        return {
            "full_image": self.full_image,
            "wrist_image": self.wrist_image,
            "state": self.proprio,
            "task_description": self.instruction,
        }


class LiberoPreprocessor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def _resize_rgb(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")

        if image.shape[-1] == 4:
            image = image[..., :3]

        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")

        pil = Image.fromarray(image)
        pil = pil.resize((self.image_size, self.image_size))
        return np.asarray(pil, dtype=np.uint8)

    def _first_existing_key(self, obs: Dict[str, Any], candidates):
        for key in candidates:
            if key in obs:
                return key
        return None

    def _quat_to_axis_angle(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [x, y, z, w] or [w, x, y, z] approximately to axis-angle.
        LIBERO/robosuite commonly uses quaternion robot state.
        """
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)

        if quat.shape[0] != 4:
            raise ValueError(f"Expected quaternion shape [4], got {quat.shape}")

        # Heuristic: robosuite often gives [x, y, z, w]
        x, y, z, w = quat

        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float32)

        x, y, z, w = quat / norm

        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        s = np.sqrt(max(1.0 - w * w, 0.0))

        if s < 1e-8:
            axis = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            axis = np.array([x / s, y / s, z / s], dtype=np.float32)

        return (axis * angle).astype(np.float32)

    def _extract_proprio(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Returns LIBERO/OpenVLA-style 8D proprio if possible:
        eef_pos [3] + eef_axis_angle [3] + gripper_qpos [2] = 8
        """
        if "state" in obs:
            return np.asarray(obs["state"], dtype=np.float32).reshape(-1)

        if all(k in obs for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]):
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
            eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)

            axis_angle = self._quat_to_axis_angle(eef_quat)

            proprio = np.concatenate([eef_pos, axis_angle, gripper], axis=0)
            return proprio.astype(np.float32)

        # Fallback: try common state-like keys
        for key in ["robot_state", "proprio", "proprio_state"]:
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32).reshape(-1)

        raise KeyError(
            "Could not extract proprio/state. Available keys: "
            f"{list(obs.keys())}"
        )

    def from_openvla_sample(self, obs: Dict[str, Any]) -> LiberoModelInput:
        """
        For OpenVLA-OFT sample_libero_spatial_observation.pkl.
        Expected keys:
        full_image, wrist_image, state, task_description
        """
        return LiberoModelInput(
            full_image=self._resize_rgb(obs["full_image"]),
            wrist_image=self._resize_rgb(obs["wrist_image"]),
            proprio=np.asarray(obs["state"], dtype=np.float32).reshape(-1),
            instruction=str(obs["task_description"]),
        )

    def from_env_obs(
        self,
        obs: Dict[str, Any],
        instruction: str,
    ) -> LiberoModelInput:
        """
        For live LIBERO env observations.
        Handles common LIBERO/robosuite image keys.
        """
        full_key = self._first_existing_key(
            obs,
            [
                "full_image",
                "agentview_image",
                "agentview_rgb",
                "front_image",
                "image",
            ],
        )

        wrist_key = self._first_existing_key(
            obs,
            [
                "wrist_image",
                "robot0_eye_in_hand_image",
                "eye_in_hand_image",
                "robot0_eye_in_hand_rgb",
            ],
        )

        if full_key is None:
            raise KeyError(f"Could not find full image key. Available keys: {list(obs.keys())}")

        if wrist_key is None:
            raise KeyError(f"Could not find wrist image key. Available keys: {list(obs.keys())}")

        full_image = self._resize_rgb(obs[full_key])
        wrist_image = self._resize_rgb(obs[wrist_key])
        proprio = self._extract_proprio(obs)

        return LiberoModelInput(
            full_image=full_image,
            wrist_image=wrist_image,
            proprio=proprio,
            instruction=instruction,
        )


def action_chunk_to_array(action_chunk: Any) -> np.ndarray:
    """
    Converts OpenVLA-OFT action chunk list into [T, 7] float32 array.
    """
    return np.asarray(action_chunk, dtype=np.float32)
