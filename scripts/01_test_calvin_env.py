import os
import time
from pathlib import Path
import numpy as np

# Compatibility patch for old CALVIN / TACTO / urdfpy code
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool


def main():
    calvin_root = Path(
        os.environ.get(
            "CALVIN_ROOT",
            "/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/calvin",
        )
    )

    assert calvin_root.exists(), f"CALVIN_ROOT does not exist: {calvin_root}"

    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    conf_dir = calvin_root / "calvin_env" / "conf"
    assert conf_dir.exists(), f"Missing CALVIN config dir: {conf_dir}"

    print(f"Using CALVIN_ROOT: {calvin_root}")
    print(f"Using config dir: {conf_dir}")

    # Clear Hydra state in case another script initialized it before
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config_data_collection",
            overrides=["cameras=static_and_gripper"],
        )

    # Important runtime settings
    cfg.env["show_gui"] = True
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True

    # Try EGL first because your log showed EGL/NVIDIA was working
    cfg.env["use_egl"] = False

    print("Instantiating CALVIN env with cameras=static_and_gripper ...")
    env = hydra.utils.instantiate(cfg.env)

    obs = env.reset()

    print("Observation keys:", obs.keys())
    print("RGB keys:", obs["rgb_obs"].keys())
    print("Robot obs shape:", obs["robot_obs"].shape)
    print("Environment reset successful.")
    print("Observation type:", type(obs))

    if isinstance(obs, dict):
        print("Observation keys:", obs.keys())

    for i in range(50):
        action = env.action_space.sample() * 1
        action[-1] = 0.0
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward}, done={done}")
        time.sleep(0.1)

    print("CALVIN Hydra env test finished successfully.")


if __name__ == "__main__":
    main()