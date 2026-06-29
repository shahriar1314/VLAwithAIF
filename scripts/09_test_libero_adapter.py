import sys
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_ROOT = PROJECT_ROOT / "external" / "openvla-oft"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"

sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(OPENVLA_ROOT))

from aif_vla.libero_adapter import LiberoPreprocessor


def save_image(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def test_openvla_sample():
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

    print("Instruction:", model_input.instruction)
    print("full_image:", model_input.full_image.shape, model_input.full_image.dtype)
    print("wrist_image:", model_input.wrist_image.shape, model_input.wrist_image.dtype)
    print("proprio:", model_input.proprio.shape, model_input.proprio.dtype)

    openvla_obs = model_input.to_openvla_observation()
    print("OpenVLA observation keys:", openvla_obs.keys())

    out_dir = PROJECT_ROOT / "outputs" / "libero_adapter_test"
    save_image(model_input.full_image, out_dir / "sample_full_image.png")
    save_image(model_input.wrist_image, out_dir / "sample_wrist_image.png")

    print("Sample observation adapter test OK.")


def test_live_libero_env():
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

    print("Task description:", task_description)
    print("Raw obs keys:", obs.keys())

    preprocessor = LiberoPreprocessor(image_size=224)
    model_input = preprocessor.from_env_obs(obs, task_description)

    print("Instruction:", model_input.instruction)
    print("full_image:", model_input.full_image.shape, model_input.full_image.dtype)
    print("wrist_image:", model_input.wrist_image.shape, model_input.wrist_image.dtype)
    print("proprio:", model_input.proprio.shape, model_input.proprio.dtype)

    out_dir = PROJECT_ROOT / "outputs" / "libero_adapter_test"
    save_image(model_input.full_image, out_dir / "live_full_image.png")
    save_image(model_input.wrist_image, out_dir / "live_wrist_image.png")

    env.close()

    print("Live LIBERO env adapter test OK.")


def main():
    test_openvla_sample()
    test_live_libero_env()

    print("\nLIBERO adapter test finished successfully.")


if __name__ == "__main__":
    main()
