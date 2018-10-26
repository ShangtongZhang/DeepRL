#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} ps -a | awk '{ print $1,$2 }' | grep deeprl | awk '{print $1 }' | xargs -I {} ${cmd} stop {}