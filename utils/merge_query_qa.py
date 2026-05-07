"""Merge per-shard query QA outputs into a single final JSON.

Usage:
    PYTHONPATH=. python utils/merge_query_qa.py mevis
    PYTHONPATH=. python utils/merge_query_qa.py revos

What it merges:
    Looks under output_query_qa/ for files of the form
        <dataset>_query_qa_shard<S>.json
    (written by each sharded invocation of Ovis_preprocess_query_<dataset>.py).
    Also falls back to per-pid worker checkpoints
        <dataset>_result_shard<S>_pid<P>.json
    if shard finals aren't found.

    Aggregates all entries (keys of form 'video/exp_id') into one dict and
    saves to output_query_qa/<dataset>_query_qa.json (no shard suffix).

Notes:
    - When two shard files contain the same key, the later-loaded one wins.
      Sharding by stride should make keys disjoint, so this only matters if
      sharding was misconfigured.
    - Run after every shard invocation has finished writing its output.
"""
import argparse
import glob
import json
import os
import sys


OUTPUT_DIR = "./output_query_qa"


def merge(dataset: str) -> None:
    base = OUTPUT_DIR
    if not os.path.isdir(base):
        print(f"[merge] error: {base} not found", file=sys.stderr)
        sys.exit(1)

    shard_finals = sorted(glob.glob(os.path.join(base, f"{dataset}_query_qa_shard*.json")))
    if shard_finals:
        sources = shard_finals
        print(f"[merge] found {len(sources)} shard final(s)")
    else:
        # fallback: per-pid worker checkpoints
        sources = sorted(glob.glob(os.path.join(base, f"{dataset}_result_shard*_pid*.json")))
        if not sources:
            # very old layout, no shard prefix
            sources = sorted(glob.glob(os.path.join(base, f"{dataset}_result_*.json")))
        if not sources:
            print(f"[merge] no shard files for dataset='{dataset}' in {base}", file=sys.stderr)
            sys.exit(1)
        print(f"[merge] no shard finals; falling back to {len(sources)} worker checkpoint(s)")

    merged: dict = {}
    for src in sources:
        try:
            with open(src) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[merge] skipping {src}: {e.__class__.__name__}: {e}", file=sys.stderr)
            continue
        before = len(merged)
        if isinstance(data, dict):
            merged.update(data)
        else:
            print(f"[merge] skipping {src}: expected dict, got {type(data).__name__}", file=sys.stderr)
            continue
        added = len(merged) - before
        print(f"[merge] {os.path.basename(src):60s}  +{added:5d}  (total {len(merged)})")

    out = os.path.join(base, f"{dataset}_query_qa.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"\n[merge] wrote {len(merged)} entries to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["mevis", "revos"], help="which dataset to merge")
    args = ap.parse_args()
    merge(args.dataset)


if __name__ == "__main__":
    main()
