#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU=$1 ${cmd} run --rm -v `pwd`:/workspace/DeepRL --entrypoint '/bin/sh' deeprl -c "python3 $2"