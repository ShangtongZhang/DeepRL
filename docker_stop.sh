#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

${cmd} ps -a | awk '{ print $1,$2 }' | grep deep_rl/v1.1 | awk '{print $1 }' | xargs -I {} ${cmd} kill {}
${cmd} ps -a | awk '{ print $1,$2 }' | grep deep_rl/v1.1 | awk '{print $1 }' | xargs -I {} ${cmd} rm {}
