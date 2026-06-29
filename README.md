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

# Random Information

## OpenVLA-OFT test status:
- Tested with moojink/openvla-7b-oft-finetuned-libero-spatial.
- Full precision fails on RTX 4060 8 GB due to CUDA OOM.
- 4-bit loading works after patching openvla_utils.py to avoid .to(device) for quantized models.
- Generated an 8-step, 7D action chunk from sample LIBERO observation.