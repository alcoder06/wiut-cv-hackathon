"""Answers from the labelling page -> dev/labels.json (the organizers' ground-truth format).

    python scripts/export_labels.py <dump dir> [--site out/label_site] [--json dev/labels/incoming/*.json]

<dump dir> holds the page's database as JSON files, one per document: votes/<id>.json
(one per candidate and person) and missed/<id>.json (events the team added). Each person
answers on their own; a candidate is real when more people said real than not real (a
tie stays out of the key: an unsure label hurts more than a missing one). Its times are
the median of the corrections people made, or the system's when nobody corrected them.
Same-class overlaps are merged into one segment, as the FAQ asks. --json adds the
files teammates download from their labelling pages ("Download answers"); one person's
answer that arrives both as a file and in a page's database counts once (latest wins).
Also writes
dev/labels/reviewed.json: every candidate with everyone's answers and reasons, which says
which classes were fully checked and why rules misfired.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.segments import merge_intervals  # noqa: E402


def load_dir(path: Path) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))} if path.exists() else {}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--site", default="out/label_site")
    ap.add_argument("--out", default="dev/labels.json")
    ap.add_argument("--json", nargs="*", default=[], help="answer files downloaded from the labelling pages")
    ap.add_argument("--skip", nargs="*", default=["0924.mp4"],
                    help="videos never written to the key (0924 is a scaled-down copy, not an official sample)")
    ap.add_argument("--all-videos", action="store_true",
                    help="also write videos that still have unanswered candidates (default: leave them out, "
                         "because the tuner scores every unlabelled detection as a false alarm)")
    args = ap.parse_args()

    site = json.loads((Path(args.site) / "candidates.json").read_text(encoding="utf-8"))
    latest: dict = {}                                       # (candidate, person) -> newest answer
    def add_vote(v: dict) -> None:
        key = (v["cid"], v["by"])
        if key not in latest or (v.get("at") or "") > (latest[key].get("at") or ""):
            latest[key] = v
    for v in load_dir(Path(args.dump) / "votes").values():
        add_vote(v)
    for f in args.json:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        who = doc.get("person_id") or "json:" + (doc.get("labeller") or Path(f).stem)
        for a in doc.get("answers", []):
            if a.get("verdict"):
                add_vote({**a, "by": who})
        print(f"{f}: {len(doc.get('answers', []))} answers from {doc.get('labeller') or who}")
    votes: dict = defaultdict(list)
    for v in latest.values():
        votes[v["cid"]].append(v)
    missed = load_dir(Path(args.dump) / "missed")

    # a round-2 "re-time" item asked for better times on an event already confirmed real; it
    # replaces its original only when judged real. A "not real" there is flagged, not applied:
    # the re-time clip shows the new detection, which can be a different moment or vehicle.
    def majority(vs):
        n = Counter(v.get("verdict") for v in vs)
        return "real" if n["real"] > n["not_real"] else "not_real" if n["not_real"] > n["real"] else None
    superseded, conflicts = set(), []
    for c in site["candidates"]:
        if c.get("retime_of") and votes.get(c["id"]):
            if majority(votes[c["id"]]) == "real":
                superseded.add(c["retime_of"])
            else:
                conflicts.append((c["id"], c["retime_of"]))
    segs: dict = defaultdict(lambda: defaultdict(list))   # video -> label -> [(s, e)]
    reviewed = []
    tally: dict = defaultdict(Counter)
    for c in site["candidates"]:
        vs = votes.get(c["id"], [])
        n = Counter(v.get("verdict") for v in vs)
        verdict = ("todo" if not vs else "real" if n["real"] > n["not_real"]
                   else "not_real" if n["not_real"] > n["real"] else "split" if n["real"] else "unsure")
        tally[c["label"]][verdict] += 1
        real = [v for v in vs if v.get("verdict") == "real"]
        starts = [v["start"] for v in real if v.get("start") is not None]
        ends = [v["end"] for v in real if v.get("end") is not None]
        s = statistics.median(starts) if starts else c["start"]
        e = statistics.median(ends) if ends else c["end"]
        if verdict == "real" and c["id"] not in superseded and not (c.get("retime_of") and c["retime_of"] not in superseded):
            segs[c["video"]][c["label"]].append((s, e))
        reviewed.append({"id": c["id"], "video": c["video"], "label": c["label"], "verdict": verdict,
                         "votes": {v["by"]: v.get("verdict") for v in vs},
                         "proposed": [c["start"], c["end"]], "times": [s, e], "current": c["current"],
                         "round": c.get("round", 1), "retime_of": c.get("retime_of"),
                         "superseded": c["id"] in superseded,
                         "reasons": [v["reason"] for v in vs if v.get("reason")],
                         "notes": [v["note"] for v in vs if v.get("note")]})
    for m in missed.values():
        segs[m["video"]][m["label"]].append((m["start"], m["end"]))
        tally[m["label"]]["missed"] += 1

    # a video goes into the key only when every candidate in it has an answer; a re-time item
    # left unanswered only keeps its round-1 times, so it doesn't hold a video back
    open_ = defaultdict(int)
    for c in site["candidates"]:
        if not c.get("retime_of") and not votes.get(c["id"]):
            open_[c["video"]] += 1
    gt, skipped = {}, {}
    for video, meta in site["videos"].items():
        if video in args.skip:
            continue
        if open_[video] and not args.all_videos:
            skipped[video] = open_[video]
            continue
        events = [[round(s, 3), round(min(e, meta["duration"]), 3), label]
                  for label, ivs in segs[video].items() for s, e in merge_intervals(ivs, 0.0)]
        gt[video] = {"duration": meta["duration"], "fps": meta["fps"], "events": sorted(events)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, indent=1), encoding="utf-8")
    # account ids are private: the committed review names people "labeller 1", "labeller 2", ...
    alias: dict = {}
    for r in reviewed:
        r["votes"] = {alias.setdefault(k, f"labeller {len(alias) + 1}"): v for k, v in r["votes"].items()}
    rev = out.parent / "labels" / "reviewed.json"
    rev.parent.mkdir(parents=True, exist_ok=True)
    rev.write_text(json.dumps(reviewed, indent=1), encoding="utf-8")

    people = {v["by"] for vs in votes.values() for v in vs}
    print(f"answers from {len(people)} people")
    print(f"{'class':<20}{'cands':>6}{'real':>6}{'not':>6}{'split':>6}{'?':>5}{'todo':>6}{'missed':>8}")
    for label, t in sorted(tally.items()):
        n = sum(v for k, v in t.items() if k != "missed")
        print(f"{label:<20}{n:>6}{t['real']:>6}{t['not_real']:>6}{t['split']:>6}{t['unsure']:>5}{t['todo']:>6}{t['missed']:>8}")
    for video, entry in gt.items():
        print(f"{video}: complete, {len(entry['events'])} events -> {out}")
    for video, n in skipped.items():
        print(f"{video}: LEFT OUT, {n} candidates still unanswered")
    for rid, orig in conflicts:
        print(f"re-time {rid} judged not real, round-1 {orig} kept as real (flagged in reviewed.json)")
    print(f"wrote {rev}")


if __name__ == "__main__":
    main()
