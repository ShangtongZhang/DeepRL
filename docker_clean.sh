#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} run --rm -v `pwd`:/shaang/DeepRL --entrypoint '/bin/sh' deep_rl/v1.1 -c "rm -rf log tf_log *.out"