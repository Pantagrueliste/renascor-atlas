#!/usr/bin/env bash
# Ship the raw files of collections to Hypatia: scripts/push_coll.sh [-n] [--siblings] [--extra GLOB] <coll-or-prefix>...
#   <coll-or-prefix>  a scan.jsonl 'coll' value or a path prefix relative to RQ/data/raw (same as COLLS in a module)
#   --siblings        also non-XML files next to the XML files (register csv/json outside archives)
#   --extra GLOB      extra glob relative to the raw root (repeatable)
#   -n                dry run (list + sizes, no transfer)
# All files of the collection are sent (registers are often separate files), whole archives for zip/tar members.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${RATLAS_RAW:-/Users/clem/Projects/Papers/RQ/data/raw}"
REMOTE="${RATLAS_REMOTE:-hypatia}"
DEST="${RATLAS_REMOTE_RAW:-renascor-atlas-data/raw}"   # relative to the remote home
DRY=0; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=1; shift;;
    --siblings) ARGS+=(--siblings); shift;;
    --extra) ARGS+=(--extra "$2"); shift 2;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) ARGS+=("$1"); shift;;
  esac
done
[ ${#ARGS[@]} -gt 0 ] || { echo "usage: $0 [-n] [--siblings] [--extra GLOB] <coll-or-prefix>..." >&2; exit 2; }
LIST="$(mktemp "${TMPDIR:-/tmp}/ratlas-push.XXXXXX")"
trap 'rm -f "$LIST"' EXIT
nice -n 19 "$REPO/.venv/bin/python" -I -m ratlas files "${ARGS[@]}" > "$LIST"
N=$(wc -l < "$LIST" | tr -d ' ')
[ "$N" -gt 0 ] || { echo "no files matched" >&2; exit 1; }
BYTES=$(cd "$RAW" && xargs -I{} stat -f %z "{}" < "$LIST" | awk '{s+=$1} END {printf "%d", s}')
echo "push_coll: $N files, $((BYTES/1048576)) MiB -> $REMOTE:~/$DEST" >&2
[ "$DRY" = 1 ] && { cat "$LIST"; exit 0; }
ssh "$REMOTE" "mkdir -p ~/$DEST"
nice -n 19 rsync -a --partial --files-from="$LIST" "$RAW/" "$REMOTE:$DEST/"
echo "push_coll: done" >&2
