#!/bin/bash
# Resumable runner for the RenNER silver teacher runs (one "lane" = one model, jobs run in order).
#
#   nohup nice -n 19 bash scripts/ner3_silver_daemon.sh JOBS_FILE LOG_DIR [WORKERS] &
#
# JOBS_FILE: one job per line, "model|think|tag|guide|passages|out" (lines starting with # are skipped).
# A job is done when every passage id of <passages> has a line in <out> without a transport error.
# Each pass calls renner_llm_teacher.py --prompt fewshot --retry-errors (resumable). On the Ollama usage limit
# (client exit code 3) the lane waits and probes the model every 10 minutes. Stop: kill the daemon's PID (see LOG_DIR/<lane>.pid).
cd /Users/clem/Projects/Papers/renascor-atlas || exit 1
JOBS=$1; LOGS=$2; W=${3:-4}
lane=$(basename "$JOBS" .txt)
mkdir -p "$LOGS"
echo $$ > "$LOGS/$lane.pid"
log() { echo "$(date '+%F %T') $*" >> "$LOGS/$lane.runs.log"; }
probe() { curl -s -m 90 localhost:11434/api/chat -d "{\"model\":\"$1\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"think\":false,\"options\":{\"num_predict\":1}}" | grep -q '"message"'; }
remaining() {  # passages of $1 without a good line in $2
  python3 - "$1" "$2" <<'EOF'
import json, sys, os
ids = {json.loads(l)['id'] for l in open(sys.argv[1], encoding='utf8') if l.strip()}
ok = set()
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2], encoding='utf8'):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        if d.get('error') in (None, 'invalid_json'):
            ok.add(d['id'])
print(len(ids - ok))
EOF
}
fails=0
while true; do
  job=""
  while IFS='|' read -r m think tag guide passages out; do
    [[ -z "$m" || "$m" == \#* ]] && continue
    [ "$(remaining "$passages" "$out")" -gt 0 ] && { job="$m|$think|$tag|$guide|$passages|$out"; break; }
  done < "$JOBS"
  [ -z "$job" ] && { log "all jobs done"; rm -f "$LOGS/$lane.pid"; exit 0; }
  IFS='|' read -r m think tag guide passages out <<< "$job"
  until probe "$m"; do log "quota or service unavailable: waiting ($tag)"; sleep 600; done
  before=$(remaining "$passages" "$out")
  log "start $tag ($before left)"
  python3 scripts/renner_llm_teacher.py --model "$m" --think "$think" --prompt fewshot --guide "$guide" \
     --passages "$passages" --out "$out" --workers "$W" --retry-errors >> "$LOGS/$tag.log" 2>&1
  rc=$?
  after=$(remaining "$passages" "$out")
  log "end $tag rc=$rc ($after left)"
  # a pass that crashed, or that made no progress without hitting the quota, counts as a failure
  if { [ $rc -ne 0 ] && [ $rc -ne 3 ]; } || { [ $rc -eq 0 ] && [ "$after" -ge "$before" ]; }; then fails=$((fails+1)); sleep 60; else fails=0; fi
  [ $fails -ge 20 ] && { log "giving up after 20 failed passes"; rm -f "$LOGS/$lane.pid"; exit 1; }
  sleep 5
done
