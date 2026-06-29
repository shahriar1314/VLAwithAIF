import sys
from pathlib import Path
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = PROJECT_ROOT / "external" / "openvla-oft"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"

sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(OPENVLA_ROOT))

from aif_vla.libero_adapter import LiberoPreprocessor
from aif_vla.openvla_oft_adapter import OpenVLAOFTAdapter


def test_sample_observation(adapter: OpenVLAOFTAdapter):
    print("\n=== Test 1: OpenVLA-OFT sample observation ===")

    sample_path = (
        OPENVLA_ROOT
        / "experiments"
        / "robot"
        / "libero"
        / "sample_libero_spatial_observation.pkl"
    )

    with open(sample_path, "rb") as f:
        obs = pickle.load(f)

    preprocessor = LiberoPreprocessor(image_size=224)
    model_input = preprocessor.from_openvla_sample(obs)

    action_chunk = adapter.predict_action_chunk(model_input)

    print("Instruction:", model_input.instruction)
    print("full_image:", model_input.full_image.shape, model_input.full_image.dtype)
    print("wrist_image:", model_input.wrist_image.shape, model_input.wrist_image.dtype)
    print("proprio:", model_input.proprio.shape, model_input.proprio.dtype)
    print("action_chunk:", action_chunk.shape, action_chunk.dtype)
    print(action_chunk)

    out_dir = PROJECT_ROOT / "outputs" / "openvla_oft_adapter_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "sample_action_chunk.npy", action_chunk)
    np.savetxt(out_dir / "sample_action_chunk.txt", action_chunk, fmt="%.6f")

    print("Sample OpenVLA-OFT adapter test OK.")


def test_live_env_observation(adapter: OpenVLAOFTAdapter):
    print("\n=== Test 2: Live LIBERO env reset observation ===")

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env

    task_suite = benchmark.get_benchmark("libero_spatial")()
    task = task_suite.get_task(0)

    env, task_description = get_libero_env(
        task,
        "openvla",
        resolution=224,
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    preprocessor = LiberoPreprocessor(image_size=224)
    model_input = preprocessor.from_env_obs(obs, task_description)

    action_chunk = adapter.predict_action_chunk(model_input)

    print("Instruction:", model_input.instruction)
    print("full_image:", model_input.full_image.shape, model_input.full_image.dtype)
    print("wrist_image:", model_input.wrist_image.shape, model_input.wrist_image.dtype)
    print("proprio:", model_input.proprio.shape, model_input.proprio.dtype)
    print("action_chunk:", action_chunk.shape, action_chunk.dtype)
    print(action_chunk)

    out_dir = PROJECT_ROOT / "outputs" / "openvla_oft_adapter_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "live_action_chunk.npy", action_chunk)
    np.savetxt(out_dir / "live_action_chunk.txt", action_chunk, fmt="%.6f")

    env.close()

    print("Live OpenVLA-OFT adapter test OK.")


def main():
    adapter = OpenVLAOFTAdapter()

    test_sample_observation(adapter)
    test_live_env_observation(adapter)

    print("\nOpenVLA-OFT adapter test finished successfully.")


if __name__ == "__main__":
    main()
