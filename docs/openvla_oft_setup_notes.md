# OpenVLA-OFT working setup notes

Tested on RTX 4060 8 GB.

Full precision loading failed with CUDA OOM.

4-bit loading worked after installing compatible bitsandbytes/accelerate setup and running:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBERO_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/LIBERO
export OPENVLA_OFT_ROOT=/home/roolab/shs/RoboLabProjects/VLAwithAIF/external/openvla-oft
export PYTHONPATH=$LIBERO_ROOT:$OPENVLA_OFT_ROOT
python scripts/05_test_openvla_oft.py

