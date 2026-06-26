import os
from pathlib import Path
import numpy as np

# Compatibility patch for old CALVIN / NumPy code
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from aif_vla.calvin_adapter import CalvinPreprocessor


def make_calvin_env():
    calvin_root = Path(os.environ["CALVIN_ROOT"])

    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = calvin_root / "calvin_env" / "conf"

    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config_data_collection",
            overrides=["cameras=static_and_gripper"],
        )

    cfg.env["show_gui"] = False
    cfg.env["use_egl"] = True
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True

    env = hydra.utils.instantiate(cfg.env)
    return env


def main():
    env = make_calvin_env()
    obs = env.reset()

    preprocessor = CalvinPreprocessor(
        image_size=224,
        camera_key="rgb_static",
    )

    instruction = "open the drawer"

    model_input = preprocessor.obs_to_model_input(
        obs=obs,
        instruction=instruction,
    )

    print("Single input:")
    print("image:", model_input["image"].shape, model_input["image"].dtype)
    print("proprio:", model_input["proprio"].shape, model_input["proprio"].dtype)
    print("instruction:", model_input["instruction"])

    batch = preprocessor.collate([model_input, model_input])

    print("\nBatch:")
    print("image:", batch.image.shape)
    print("proprio:", batch.proprio.shape)
    print("action:", batch.action)
    print("instruction:", batch.instruction)

    print("\nCALVIN adapter live test finished successfully.")


if __name__ == "__main__":
    main()
