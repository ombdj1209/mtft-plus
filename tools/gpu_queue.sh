#!/usr/bin/env bash
# Sequential GPU job queue: runs the first line of gpu_queue.txt, removes it, repeats. Blank line / '#' = skip.
# A line "STOP" ends the runner. Jobs append their output to logs/queue/<n>.log; a summary goes to logs/queue/queue.log.
# Fix over gpu_queue.sh: the job's exit code is captured before $(date) runs (gpu_queue.sh always logged "exit 0").
cd "$(dirname "$0")"
mkdir -p logs/queue
touch gpu_queue.txt
n=$(ls logs/queue/*.log 2>/dev/null | grep -c '/[0-9]*\.log$')
while true; do
  line=$(head -n 1 gpu_queue.txt)
  if [ -z "$line" ] && [ ! -s gpu_queue.txt ]; then sleep 30; continue; fi
  tail -n +2 gpu_queue.txt > gpu_queue.tmp && mv gpu_queue.tmp gpu_queue.txt
  case "$line" in ""|\#*) continue;; STOP) echo "$(date '+%F %T') STOP" >> logs/queue/queue.log; exit 0;; esac
  n=$((n+1))
  echo "$(date '+%F %T') START [$n] $line" >> logs/queue/queue.log
  bash -c "$line" > "logs/queue/$n.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') END   [$n] exit $rc ($line)" >> logs/queue/queue.log
done
