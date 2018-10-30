#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} run --rm -v `pwd`:/workspace/DeepRL --entrypoint '/bin/sh' deeprl -c "rm -rf log tf_log *.out"