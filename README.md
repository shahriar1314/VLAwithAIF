# Running the Project

This project uses two environments:

* `calvin_env` for running and testing the CALVIN simulation environment
* `openvla-oft` for running OpenVLA-OFT related code

## 1. Go to the project root

Always run the project scripts from the main project directory:

```bash
cd ~/shs/RoboLabProjects/VLAwithAIF
```

The scripts are located in:

```bash
scripts/
```


## 2. Activate the CALVIN environment

```bash
source external/calvin/calvin_env/bin/activate
```

After activation, the terminal should show:

```bash
(calvin_env)
```

## 3. Test the CALVIN environment

Location: 

```bash
external/calvin
```

Run:

```bash
python scripts/01_test_calvin_env.py
```



## 4. Activate OpenVLA-OFT environment

For OpenVLA-OFT related code, use:

```bash
conda activate openvla-oft
```

Use this environment when running OpenVLA-OFT models, fine-tuning scripts, or inference code.

<br>
<br>

# How to Run the Scripts

## Script 5: Testing OpenVLA-OFT

```bash

cd /home/roolab/shs/RoboLabProjects/VLAwithAIF

conda activate openvla-oft

unset PYTHONPATH
export LIBERO_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/LIBERO
export OPENVLA_OFT_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/05_test_openvla_oft.py
```

## Script 6: Visualization OpenVLA-OFT sample 

```bash
cd /home/roolab/shs/RoboLabProjects/VLAwithAIF

conda activate openvla-oft

unset PYTHONPATH
export LIBERO_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/LIBERO
export OPENVLA_OFT_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/06_visualize_openvla_oft_sample.py

```

<br>

## Script X : running libero evaluatin script: 

Run this small simulator-only test:

```bash

cd /home/roolab/shs/RoboLabProjects/VLAwithAIF

unset PYTHONPATH
export LIBERO_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/LIBERO
export OPENVLA_OFT_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT

python - <<'PY'
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

task_suite = benchmark.get_benchmark("libero_spatial")()
task = task_suite.get_task(0)

env, task_description = get_libero_env(task, "openvla", resolution=224)
print("Task:", task_description)

obs = env.reset()
print("Reset OK")
print("Obs keys:", obs.keys())

env.close()
print("LIBERO env test finished successfully.")
PY

```
Once it OK, then run: 

```bash
cd /home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft

conda activate openvla-oft

unset PYTHONPATH
export LIBERO_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/LIBERO
export OPENVLA_OFT_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --load_in_4bit True \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True
  ```

<br>

# Script 7: Running the Baseline OpenVLA-OFT

For first reproducible check, use 1 trial per task:
```bash
cd /home/roolab/shs/RoboLabProjects/VLAwithAIF
conda activate openvla-oft

BASELINE_TRIALS=1 bash scripts/07_run_libero_spatial_baseline.sh
```
For Stronger baseline increase trial number

<br>
<br>





# Random Information

## OpenVLA-OFT test status:
- Tested with moojink/openvla-7b-oft-finetuned-libero-spatial.
- Full precision fails on RTX 4060 8 GB due to CUDA OOM.
- 4-bit loading works after patching openvla_utils.py to avoid .to(device) for quantized models.
- Generated an 8-step, 7D action chunk from sample LIBERO observation.