#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU=$1 ${cmd} run --rm -v `pwd`:/shaang/DeepRL --entrypoint '/bin/sh' deep_rl/v1.1 -c "OMP_NUM_THREADS=1 python3 $2"