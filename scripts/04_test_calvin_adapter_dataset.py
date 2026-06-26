import os
from pathlib import Path

from aif_vla.calvin_adapter import CalvinPreprocessor, load_npz_episode


def main():
    calvin_root = Path(os.environ["CALVIN_ROOT"])
    dataset_dir = calvin_root / "dataset" / "calvin_debug_dataset" / "training"

    episode_files = sorted(dataset_dir.glob("episode_*.npz"))

    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {dataset_dir}")

    episode_path = episode_files[0]
    print(f"Using episode: {episode_path}")

    data = load_npz_episode(str(episode_path))

    print("\nDataset keys:")
    for key, value in data.items():
        print(f"{key}: shape={getattr(value, 'shape', None)}, dtype={getattr(value, 'dtype', None)}")

    preprocessor = CalvinPreprocessor(
        image_size=224,
        camera_key="rgb_static",
    )

    instruction = "open the drawer"

    item = preprocessor.dataset_timestep_to_model_input(
        data=data,
        t=0,
        instruction=instruction,
        action_key="rel_actions",
    )

    print("\nSingle dataset timestep:")
    print("image:", item["image"].shape, item["image"].dtype)
    print("proprio:", item["proprio"].shape, item["proprio"].dtype)
    print("instruction:", item["instruction"])

    if "action" in item:
        print("action:", item["action"].shape, item["action"].dtype)
    else:
        print("action: not found")

    batch = preprocessor.collate([item, item])

    print("\nBatch:")
    print("image:", batch.image.shape)
    print("proprio:", batch.proprio.shape)
    print("action:", None if batch.action is None else batch.action.shape)
    print("instruction:", batch.instruction)

    print("\nCALVIN adapter dataset test finished successfully.")


if __name__ == "__main__":
    main()
