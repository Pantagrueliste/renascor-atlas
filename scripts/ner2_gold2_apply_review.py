"""Apply the owner's review (artifact db "reviews") to proposed_gold_v2.jsonl and rescore the systems.

Usage: python3 scripts/ner2_gold2_apply_review.py <reviews dir> <gold2 dir> [owner rules.json]

The review page stores, per passage, only the spans whose decision the owner set
(sp: {"start-end": "y"|"n"|"u"}), the spans the owner added (add: [[start, end]]) and a
"done" flag. Spans the owner left alone keep the proposal: agreed/proposed -> place,
rejected -> not a place, to_check -> doubt.

Accepted spans that overlap (boundary variants the owner accepted both, e.g.
"Mildenhall" and "Mildenhall River") form one entity with alternatives: a prediction
equal to any alternative is one true positive, and missing all of them is one false
negative. Doubt spans are neutral in scoring.

Writes final_gold_v2.jsonl, scores_final.json and scores_final.md in the gold2 dir.
"""
import collections, glob, json, os, sys

SYS = ["r3_mdeberta_HM", "r3_mmbert_base_HM", "r3_xlmr_large_H", "r3_hmteams_HM", "flair_ner_english_large", "llm_qwen35_9b"]
ENC = SYS[:4]
LABEL = {"r3_mdeberta_HM": "mDeBERTa", "r3_mmbert_base_HM": "mmBERT", "r3_xlmr_large_H": "XLM-R large",
         "r3_hmteams_HM": "hmTEAMS", "flair_ner_english_large": "flair EN", "llm_qwen35_9b": "Qwen3.5-9B (LLM)",
         "vote_ge2": "vote ≥ 2 encodeurs", "vote_ge2_conf98": "vote ≥ 2 encodeurs et conf ≥ 0.98",
         "union_enc": "union des 4 encodeurs"}
LANGS = ["en", "la", "it", "es", "fr", "de"]
NAMES = {"en": "anglais", "la": "latin", "it": "italien", "es": "espagnol", "fr": "français", "de": "allemand"}
DEFAULT = {"agreed": "y", "proposed": "y", "rejected": "n", "to_check": "u"}
FINAL = {"y": "place", "n": "not_place", "u": "doubt"}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def groups(spans):
    """Union overlapping (start, end) pairs into groups of alternatives."""
    out = []
    for s in sorted(spans):
        hit = [g for g in out if any(overlaps(s, t) for t in g)]
        merged = {s}.union(*hit) if hit else {s}
        out = [g for g in out if g not in hit] + [merged]
    return [sorted(g) for g in out]


def apply_review(prop, rev, rules=None):
    """Return the final record for one passage and a Counter of how decisions were made.

    rules maps (start, end) to (decision, reason): owner rules applied after the review,
    e.g. "a place inside a personal name is not a place".
    """
    sp, add = rev.get("sp", {}), [tuple(x) for x in rev.get("add", [])]
    rules = rules or {}
    c = collections.Counter()
    spans = []
    known = set()
    for s in prop["spans"]:
        k = (s["start"], s["end"])
        known.add(k)
        d0 = DEFAULT[s["status"]]
        d = sp.get(f"{k[0]}-{k[1]}", d0)
        by = "owner_changed" if d != d0 else ("owner_confirmed" if f"{k[0]}-{k[1]}" in sp else "proposal")
        if k in rules and rules[k][0] != d:
            d, by = rules[k][0], "owner_rule"
            s = {**s, "rule_reason": rules[k][1]}
        c[(s["status"], FINAL[d], by)] += 1
        spans.append({**s, "proposed_status": s["status"], "status": FINAL[d], "decided_by": by})
    toks, starts = prop["tokens"], prop["token_starts"]
    for a, b in add:
        if (a, b) in known:
            continue
        cs, ce = starts[a], starts[b - 1] + len(toks[b - 1])
        spans.append({"span_id": f"{prop['id']}#add{a}-{b}", "start": a, "end": b, "text": " ".join(toks[a:b]),
                      "surface": prop["text"][cs:ce], "char_start": cs, "char_end": ce, "proposed_status": None,
                      "status": "place", "decided_by": "owner_added", "sources": {}, "reason": "ajouté par le propriétaire"})
        c[(None, "place", "owner_added")] += 1
    places = [(s["start"], s["end"]) for s in spans if s["status"] == "place"]
    gid = {}
    for i, g in enumerate(groups(places)):
        for k in g:
            gid[k] = i
    for s in spans:
        s["entity"] = gid.get((s["start"], s["end"]))
    rec = {k: v for k, v in prop.items() if k != "spans"}
    rec["spans"] = sorted(spans, key=lambda s: (s["start"], s["end"]))
    rec["reviewed"] = bool(rev.get("done"))
    rec["review_note"] = rev.get("note") or ""
    return rec, c


def predictions(pr, name):
    if name in SYS:
        return {(s["s"], s["e"]) for s in pr["models"].get(name, [])}
    votes = collections.defaultdict(list)
    for m in ENC:
        for s in pr["models"].get(m, []):
            votes[(s["s"], s["e"])].append(s.get("conf") or 0)
    if name == "union_enc":
        return set(votes)
    minconf = 0.98 if name == "vote_ge2_conf98" else 0
    return {k for k, v in votes.items() if len(v) >= 2 and max(v) >= minconf}


def score_passage(rec, pred, lenient=False):
    """Counts (tp, fp, fn) for one passage. Entities with alternatives count once; doubts are neutral."""
    ents = collections.defaultdict(set)
    for s in rec["spans"]:
        if s["status"] == "place":
            ents[s["entity"]].add((s["start"], s["end"]))
    doubt = {(s["start"], s["end"]) for s in rec["spans"] if s["status"] == "doubt"}
    match = (lambda p, alts: any(overlaps(p, a) for a in alts)) if lenient else (lambda p, alts: p in alts)
    found = set()
    tp = fp = 0
    for p in pred:
        hit = [e for e, alts in ents.items() if match(p, alts)]
        if hit:
            new = [e for e in hit if e not in found]
            if new:
                tp += 1
                found.add(new[0])
            # a second prediction on an already found entity is neither credited nor penalised
        elif p in doubt or (lenient and any(overlaps(p, d) for d in doubt)):
            continue
        else:
            fp += 1
    return tp, fp, len(ents) - len(found)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_rules(path):
    """Owner rules: a JSON list of {pid, start, end, decision y|n|u, reason}."""
    rules = collections.defaultdict(dict)
    if path:
        for r in json.load(open(path, encoding="utf8")):
            rules[r["pid"]][(r["start"], r["end"])] = (r["decision"], r["reason"])
    return rules


def main(rev_dir, gold_dir, rules_path=None):
    rules = load_rules(rules_path)
    prop = {}
    for line in open(os.path.join(gold_dir, "proposed_gold_v2.jsonl"), encoding="utf8"):
        p = json.loads(line)
        prop[p["id"]] = p
    revs = {}
    for f in glob.glob(os.path.join(rev_dir, "*.json")):
        r = json.load(open(f, encoding="utf8"))
        revs[r["pid"]] = r
    missing = sorted(set(prop) - set(revs))
    out, how = [], collections.Counter()
    for pid, p in prop.items():
        rec, c = apply_review(p, revs.get(pid, {}), rules.get(pid))
        out.append(rec)
        how.update(c)
    with open(os.path.join(gold_dir, "final_gold_v2.jsonl"), "w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pred = {}
    for line in open(os.path.join(gold_dir, "predictions.jsonl"), encoding="utf8"):
        r = json.loads(line)
        pred[r["id"]] = r
    rows = SYS + ["vote_ge2", "vote_ge2_conf98", "union_enc"]
    res = {}
    for mode in ("strict", "lenient"):
        for m in rows:
            for lang in LANGS + ["all"]:
                if m == "flair_ner_english_large" and lang != "en":
                    continue
                t = [0, 0, 0]
                for rec in out:
                    if lang != "all" and rec["lang"] != lang:
                        continue
                    for i, v in enumerate(score_passage(rec, predictions(pred[rec["id"]], m), mode == "lenient")):
                        t[i] += v
                res[f"{mode}|{lang}|{m}"] = [*prf(*t), *t]
    counts = collections.defaultdict(collections.Counter)
    for rec in out:
        c = counts[rec["lang"]]
        c["passages"] += 1
        ents = {s["entity"] for s in rec["spans"] if s["status"] == "place"}
        c["entities"] += len(ents)
        c["doubt"] += sum(s["status"] == "doubt" for s in rec["spans"])
        c["no_place"] += not ents
        c["reviewed"] += rec["reviewed"]
    json.dump({"scores": res, "counts": {k: dict(v) for k, v in counts.items()},
               "decisions": {"|".join(map(str, k)): v for k, v in how.items()}, "missing_reviews": missing},
              open(os.path.join(gold_dir, "scores_final.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)

    def table(mode, title):
        lines = [f"### {title}", "", "| Système | " + " | ".join(NAMES[l] for l in LANGS) + " | toutes langues |",
                 "|---|" + "---|" * (len(LANGS) + 1)]
        for m in rows:
            cells = []
            for lang in LANGS + ["all"]:
                v = res.get(f"{mode}|{lang}|{m}")
                cells.append("—" if v is None else f"{v[2]:.2f} ({v[0]:.2f}/{v[1]:.2f})")
            lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    tot = collections.Counter()
    for c in counts.values():
        tot.update(c)
    md = [
        "# Scores sur le jeu de référence v2 relu",
        "",
        f"Référence : `final_gold_v2.jsonl`, {tot['passages']} passages relus par le propriétaire ({tot['reviewed']} marqués relus), "
        f"{tot['entities']} lieux, {tot['doubt']} cas de doute (neutres), {tot['no_place']} passages sans lieu.",
        "",
        "Cellule : F1 (précision/rappel), micro-moyenne. Un lieu dont le propriétaire a accepté deux découpages compte une fois.",
        "",
        "| Langue | Passages | Lieux | Doutes | Sans lieu |",
        "|---|---|---|---|---|",
        *[f"| {NAMES[l]} | {counts[l]['passages']} | {counts[l]['entities']} | {counts[l]['doubt']} | {counts[l]['no_place']} |" for l in LANGS],
        "",
        table("strict", "A. Correspondance exacte"),
        "",
        table("lenient", "B. Correspondance par recouvrement (frontières tolérées)"),
        "",
    ]
    open(os.path.join(gold_dir, "scores_final.md"), "w", encoding="utf8").write("\n".join(md))
    print(dict(tot), "missing reviews:", len(missing))
    for k, v in sorted(how.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main(*sys.argv[1:4])
