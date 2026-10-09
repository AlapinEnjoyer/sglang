#!/usr/bin/env bash
# Stop any running server, launch serve.sh with the given args, wait until ready. Usage: restart.sh LOGNAME preset [args...]
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
logdir="${LOGDIR:-/tmp/claude-1000/-home-ubuntu-experiments-sglang/40087568-b731-43fd-b90c-f03af6e6fea9/scratchpad/logs}"
name="$1"; shift
pkill -f "[s]glang serve" ; pkill -f "[s]glang::" ; sleep 3
while nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{f=1} END{exit !f}'; do sleep 2; done
nohup "$here/serve.sh" "$@" > "$logdir/$name.log" 2>&1 &
"$here/wait_ready.sh" "$logdir/$name.log"
