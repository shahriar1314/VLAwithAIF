import sys
from pathlib import Path
import pickle

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_OFT_ROOT = PROJECT_ROOT / "external" / "openvla-oft"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"

sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(OPENVLA_OFT_ROOT))


from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


def save_image(array, path):
    array = np.asarray(array)

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    Image.fromarray(array).save(path)


def main():
    output_dir = PROJECT_ROOT / "outputs" / "openvla_oft_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )

    print("Loading OpenVLA-OFT model...")
    vla = get_vla(cfg)

    print("Loading processor...")
    processor = get_processor(cfg)

    print("Loading action head...")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    print("Loading proprio projector...")
    proprio_projector = get_proprio_projector(
        cfg,
        llm_dim=vla.llm_dim,
        proprio_dim=PROPRIO_DIM,
    )

    sample_obs_path = (
        OPENVLA_OFT_ROOT
        / "experiments"
        / "robot"
        / "libero"
        / "sample_libero_spatial_observation.pkl"
    )

    print(f"Loading sample observation: {sample_obs_path}")
    with open(sample_obs_path, "rb") as f:
        observation = pickle.load(f)

    instruction = observation["task_description"]

    print("Instruction:", instruction)

    save_image(observation["full_image"], output_dir / "full_image.png")
    save_image(observation["wrist_image"], output_dir / "wrist_image.png")

    print("Generating action chunk...")
    actions = get_vla_action(
        cfg,
        vla,
        processor,
        observation,
        instruction,
        action_head,
        proprio_projector,
    )

    actions = np.asarray(actions, dtype=np.float32)

    np.save(output_dir / "action_chunk.npy", actions)
    np.savetxt(output_dir / "action_chunk.txt", actions, fmt="%.6f")

    print("Action chunk shape:", actions.shape)

    labels = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        plt.plot(actions[:, i], marker="o", label=label)

    plt.title("OpenVLA-OFT predicted action chunk")
    plt.xlabel("Open-loop action step")
    plt.ylabel("Action value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "action_chunk_plot.png", dpi=200)
    plt.close()

    print("\nSaved visualization files:")
    print(output_dir / "full_image.png")
    print(output_dir / "wrist_image.png")
    print(output_dir / "action_chunk_plot.png")
    print(output_dir / "action_chunk.npy")
    print(output_dir / "action_chunk.txt")


if __name__ == "__main__":
    main()

