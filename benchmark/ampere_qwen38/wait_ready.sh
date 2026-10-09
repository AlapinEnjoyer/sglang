#!/usr/bin/env bash
# Block until the server log reports ready (exit 0) or a failure (exit 1). Usage: wait_ready.sh LOG
log="$1"
for _ in $(seq 1 360); do
  grep -q "fired up and ready" "$log" && { echo READY; exit 0; }
  grep -qE "Traceback|Error:|exited with code" "$log" && { echo FAILED; grep -E -A3 "Traceback|Error" "$log" | tail -40; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
