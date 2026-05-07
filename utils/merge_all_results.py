"""Merge per-shard inference outputs into a single all_results.json.

Usage:
    PYTHONPATH=. python utils/merge_all_results.py /home/cvlab18/media/data3/jaeho/mevis_qwen
    PYTHONPATH=. python utils/merge_all_results.py /home/cvlab18/media/data3/jaeho/revos_qwen

What it merges:
    Looks under <output_dir>/Annotations/ for files of the form
        all_results_shard<S>.json     (per-shard final, written when the
                                       sharded invocation of
                                       eval/Ovis_infer_<dataset>.py finishes)
    Falls back to per-pid worker checkpoints
        all_results_shard<S>_pid<P>.json
    if shard finals aren't found.

    Concatenates all entries (each entry is a dict with all_boxes, all_points,
    video_name, exp_id, ...) into one list and saves to
        <output_dir>/Annotations/all_results.json

Notes:
    - Entries are deduplicated by (video_name, exp_id). When duplicates exist,
      the later-loaded one wins.
    - Run after every shard invocation has finished writing its output.
    - downstream eval/post_processing.py reads
        <output_dir>/Annotations/all_results.json
"""
import argparse
import glob
import json
import os
import sys


def merge(output_dir: str) -> None:
    annot_dir = os.path.join(output_dir, "Annotations")
    if not os.path.isdir(annot_dir):
        print(f"[merge] error: {annot_dir} not found", file=sys.stderr)
        sys.exit(1)

    shard_finals = sorted(glob.glob(os.path.join(annot_dir, "all_results_shard*.json")))
    # filter out per-pid files (have '_pid' in name) — those are checkpoints, not finals
    shard_finals = [p for p in shard_finals if "_pid" not in os.path.basename(p)
                    and os.path.basename(p) != "all_results.json"]

    if shard_finals:
        sources = shard_finals
        print(f"[merge] found {len(sources)} shard final(s)")
    else:
        sources = sorted(glob.glob(os.path.join(annot_dir, "all_results_shard*_pid*.json")))
        if not sources:
            sources = sorted(glob.glob(os.path.join(annot_dir, "all_results_*.json")))
            sources = [p for p in sources if os.path.basename(p) != "all_results.json"]
        if not sources:
            print(f"[merge] no shard files in {annot_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"[merge] no shard finals; falling back to {len(sources)} worker checkpoint(s)")

    # Dedupe by (video_name, exp_id), later wins.
    by_key: dict = {}
    for src in sources:
        try:
            with open(src) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[merge] skipping {src}: {e.__class__.__name__}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"[merge] skipping {src}: expected list, got {type(data).__name__}", file=sys.stderr)
            continue
        before = len(by_key)
        for entry in data:
            key = (entry.get("video_name"), entry.get("exp_id"))
            by_key[key] = entry
        added = len(by_key) - before
        print(f"[merge] {os.path.basename(src):60s}  +{added:5d}  (total {len(by_key)})")

    merged = list(by_key.values())
    out = os.path.join(annot_dir, "all_results.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[merge] wrote {len(merged)} entries to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", help="dataset output dir (the OUTPUT path used in Ovis_infer_*.py, e.g. /home/cvlab18/media/data3/jaeho/mevis_qwen)")
    args = ap.parse_args()
    merge(args.output_dir)


if __name__ == "__main__":
    main()
