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
