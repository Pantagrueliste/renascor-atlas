"""Split the passages still to annotate into one shard per GPU slot (data-parallel silver runs).

Usage: python3 scripts/ner3_silver_shard.py SLOTS SHARD_DIR SOURCE [SOURCE ...]
SOURCE = GUIDE|PASSAGES|OUT_PREFIX. A passage is done when its id has a line in any OUT_PREFIX*.jsonl.
Writes SHARD_DIR/slotNN.tsv: one line per source with work, "GUIDE<TAB>shard file<TAB>OUT_PREFIX.sNN.jsonl".
Remaining passages of each source are dealt round-robin, so every slot gets a similar share of every source.
Consecutive sources with the same guide form one group (one model load per group and slot): their passages share one
shard file, written to the first source's OUT_PREFIX.sNN.jsonl, and "done" is checked against all prefixes of the group.
"""
import glob, json, os, sys


def done_ids(prefix):
    ids = set()
    for f in glob.glob(prefix + "*.jsonl"):
        for line in open(f, encoding="utf8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error") in (None, "invalid_json"):
                ids.add(d["id"])
    return ids


def main(slots, shard_dir, sources):
    slots = int(slots)
    os.makedirs(shard_dir, exist_ok=True)
    plan = [[] for _ in range(slots)]
    groups = []
    for src in sources:
        guide, passages, prefix = src.split("|")
        if groups and groups[-1][0] == guide:
            groups[-1][1].append((passages, prefix))
        else:
            groups.append((guide, [(passages, prefix)]))
    for k, (guide, members) in enumerate(groups):
        done = set().union(*(done_ids(p) for _, p in members))
        todo = [l for passages, _ in members for l in open(passages, encoding="utf8")
                if l.strip() and json.loads(l)["id"] not in done]
        prefix = members[0][1]
        print(f"{guide}: {len(todo)} to do, {len(done)} done ({', '.join(os.path.basename(m) for m, _ in members)})", flush=True)
        for s in range(slots):
            part = todo[s::slots]
            if not part:
                continue
            path = os.path.join(shard_dir, f"src{k:02d}_slot{s:02d}.jsonl")
            with open(path, "w", encoding="utf8") as f:
                f.writelines(part)
            plan[s].append(f"{guide}\t{path}\t{prefix}.s{s:02d}.jsonl")
    for s in range(slots):
        with open(os.path.join(shard_dir, f"slot{s:02d}.tsv"), "w", encoding="utf8") as f:
            f.write("\n".join(plan[s]) + ("\n" if plan[s] else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
