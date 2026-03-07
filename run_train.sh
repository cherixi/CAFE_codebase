#!/bin/bash
# Use conda's libstdc++ (fixes GLIBCXX_3.4.26 on older systems e.g. Ubuntu 18.04)
export LD_LIBRARY_PATH=/share/apps/anaconda3/envs/aixi_cafe_cursor/lib:${LD_LIBRARY_PATH:-}
cd "$(dirname "$0")"
python train.py "$@"