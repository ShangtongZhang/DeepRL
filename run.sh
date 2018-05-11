#!/usr/bin/env bash

function run_args_proxy() {
  # Enable host-mode networking if necessary to access proxy on localhost
  env | grep -iP '^(https?|ftp)_proxy=.*$' | grep -qP 'localhost|127\.0\.0\.1' && {
    echo -n "--network host "
  }
  # Proxy environment variables as "--build-arg https_proxy=https://..."
  env | grep -iP '^(https?|ftp|no)_proxy=.*$' | while read env_proxy; do
    echo -n "--env ${env_proxy} "
  done
}
echo $(run_args_proxy)

nvidia-docker run -v ~/workspace/DeepRL:/workspace/DeeRL $(run_args_proxy) -d -P --name deep_rl deep_rl