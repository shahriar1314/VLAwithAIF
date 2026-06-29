import sys
from pathlib import Path
import pickle
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_OFT_ROOT = PROJECT_ROOT / "external" / "openvla-oft"

# Make sure Python can import OpenVLA-OFT modules
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


def main():
    print("Testing OpenVLA-OFT only.")
    print(f"OpenVLA-OFT root: {OPENVLA_OFT_ROOT}")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    # For your RTX 4060 8GB, start with 4-bit loading.
    # If you later use a >=16GB GPU, set load_in_4bit=False.
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

    print("\nLoading VLA model...")
    vla = get_vla(cfg)
    print("Model loaded.")

    print("\nLoading processor...")
    processor = get_processor(cfg)
    print("Processor loaded.")

    print("\nLoading action head...")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    print("Action head loaded.")

    print("\nLoading proprio projector...")
    proprio_projector = get_proprio_projector(
        cfg,
        llm_dim=vla.llm_dim,
        proprio_dim=PROPRIO_DIM,
    )
    print("Proprio projector loaded.")

    sample_obs_path = OPENVLA_OFT_ROOT / "experiments" / "robot" / "libero" / "sample_libero_spatial_observation.pkl"

    if not sample_obs_path.exists():
        raise FileNotFoundError(f"Missing sample observation: {sample_obs_path}")

    print(f"\nLoading sample observation: {sample_obs_path}")
    with open(sample_obs_path, "rb") as f:
        observation = pickle.load(f)

    print("Observation keys:", observation.keys())
    print("Task description:", observation["task_description"])

    print("\nGenerating action chunk...")
    actions = get_vla_action(
        cfg,
        vla,
        processor,
        observation,
        observation["task_description"],
        action_head,
        proprio_projector,
    )

    print("\nGenerated action chunk:")
    print("type:", type(actions))
    print("length:", len(actions))

    for i, act in enumerate(actions):
        print(f"action[{i}]: {act}")

    print("\nOpenVLA-OFT test finished successfully.")


if __name__ == "__main__":
    main()