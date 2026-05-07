"""Per-sample inference time measurement (MeViS only).

Runs the full Refer-Agent pipeline on a single (video, exp) sample and reports
wall-clock time for each stage:
    1. CLIP scoring   (per-sample, no amortization across video's expressions)
    2. MLLM preprocess (convert_query + generate_qa via vLLM)
    3. Agent inference (process_one_sample — the agent loop)
    4. SAM2 mask propagation

This is a measurement-only path, completely independent of the production
sub_processor.  No checkpointing, no resume, no sharding.

Usage:
    PYTHONPATH=. python scripts/measure_per_sample.py \\
        --video <video_id> --exp <exp_id>

Optional:
    --skip-stage4    skip SAM2 propagation (useful when you don't have a GPU
                     budget for SAM2 right now).
    --max-query-num / --max-update-frame-num / --max-update-descriptions-num
                     mirror the eval/Ovis_infer_mevis.py knobs (defaults match).

Prereqs (server-side):
    - vLLM server reachable at REFER_VLLM_BASE_URL (default http://localhost:10000/v1)
    - CLIP-base model weights downloadable / cached
    - SAM2 checkpoint at sam2/checkpoints/sam2.1_hiera_large.pt (only if not --skip-stage4)
"""
import argparse
import json
import os
import sys
import time
import types
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

# project root on path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Reuse production helpers and the refactored process_one_sample.
from eval.Ovis_infer_mevis import process_one_sample
from utils.preprocess_mevis import ImageScorer
from utils.Ovis_preprocess_query_mevis import (
    convert_query_with_vlm,
    generate_qa_with_vlm,
)
from utils.vl_backend import load_vlm

# ---------------------------------------------------------------------------
# Constants / paths (mevis only)
# ---------------------------------------------------------------------------
MEVIS_ROOT = "/home/cvlab18/media/data2/datasets/mevis_v2"
MEVIS_META = os.path.join(MEVIS_ROOT, "valid", "meta_expressions.json")
MEVIS_IMG_FOLDER = os.path.join(MEVIS_ROOT, "valid", "JPEGImages")

SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ---------------------------------------------------------------------------
# vLLM call counter (instrument client to track requests per stage)
# ---------------------------------------------------------------------------

class CallCounter:
    """Wraps an OpenAI client's chat.completions.create to count requests."""

    def __init__(self, client):
        self._client = client
        self._orig = client.chat.completions.create
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        def wrapped(*args, **kwargs):
            self.calls += 1
            resp = self._orig(*args, **kwargs)
            try:
                if resp.usage is not None:
                    self.prompt_tokens += resp.usage.prompt_tokens
                    self.completion_tokens += resp.usage.completion_tokens
            except Exception:
                pass
            return resp

        client.chat.completions.create = wrapped

    def reset(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


# ---------------------------------------------------------------------------
# Stage 1: CLIP scoring (per-sample, no amortization)
# ---------------------------------------------------------------------------

def stage1_clip_score(scorer, video_name, frames):
    """Encode all frames for ONE video. Returns dict of features for re-use in
    the cosine-sim step. Time measured covers image encoding only — text encode
    is done separately for fairness.
    """
    image_paths = [os.path.join(MEVIS_IMG_FOLDER, video_name, f + ".jpg") for f in frames]
    image_feats = scorer.encode_images(image_paths)
    return image_feats


def stage1_clip_score_per_sample(scorer, video_name, frames, exp_text):
    """Full per-sample CLIP scoring: encode N images + 1 text + sim + normalize.

    Mirrors what Step 1 (preprocess_mevis.py) computes for the (video, exp)
    pair, but without amortizing image encoding across the video's expressions.
    """
    image_paths = [os.path.join(MEVIS_IMG_FOLDER, video_name, f + ".jpg") for f in frames]

    image_feats = scorer.encode_images(image_paths)               # [N, D]
    text_feats = scorer.encode_texts([exp_text])                   # [1, D]
    sim = (image_feats @ text_feats.T).cpu().numpy()               # [N, 1]

    raw = {f + ".jpg": float(sim[i, 0]) for i, f in enumerate(frames)}
    normalized = scorer.normalize_scores(raw)
    per_frame = {
        f + ".jpg": {
            "clip": round(normalized[f + ".jpg"], 2),
            "weighted_score": round(normalized[f + ".jpg"], 2),
        }
        for f in frames
    }
    return per_frame


# ---------------------------------------------------------------------------
# Stage 4: SAM2 mask propagation (single sample)
# ---------------------------------------------------------------------------

def stage4_sam2(entry):
    """Run SAM2 video predictor on the bbox/point produced by Stage 3.

    Mirrors save_results() from eval/post_processing.py minus the disk-write
    side-effect (we write to a temp dir to avoid clobbering production output).
    """
    from sam2.build_sam import build_sam2_video_predictor

    if not entry["all_boxes"]:
        return {"warning": "no boxes from agent; nothing to propagate"}

    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)

    frame_list = [
        os.path.join(entry["img_folder"], entry["video_name"], f + ".jpg")
        for f in entry["frames"]
    ]
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    inference_state = video_predictor.init_state(
        frame_list=frame_list, offload_video_to_cpu=True
    )
    key_frame_idx = entry["key_frame_idx"]
    for obj_id, (point, box) in enumerate(zip(entry["all_points"], entry["all_boxes"])):
        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=key_frame_idx,
            obj_id=obj_id,
            points=np.array([point], dtype=np.float32),
            labels=np.array([1], np.int32),
            box=box,
        )
    n_segments = 0
    for _ in video_predictor.propagate_in_video(inference_state):
        n_segments += 1
    for _ in video_predictor.propagate_in_video(inference_state, reverse=True):
        n_segments += 1
    return {"n_propagated_steps": n_segments}


# ---------------------------------------------------------------------------
# Args namespace shim
# ---------------------------------------------------------------------------

class Args(types.SimpleNamespace):
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="MeViS video id")
    ap.add_argument("--exp", required=True, help="MeViS expression id (e.g. '0','1',...)")
    ap.add_argument("--max-query-num", type=int, default=3)
    ap.add_argument("--max-update-frame-num", type=int, default=2)
    ap.add_argument("--max-update-descriptions-num", type=int, default=2)
    ap.add_argument("--skip-stage4", action="store_true", help="skip SAM2 propagation")
    args_cli = ap.parse_args()

    print("=" * 70)
    print(f"Per-sample measurement: video={args_cli.video} exp={args_cli.exp}")
    print("=" * 70)

    # ---- load metadata ----------------------------------------------------
    print("\n[setup] loading meta_expressions.json ...")
    meta_full = json.load(open(MEVIS_META))["videos"]
    if args_cli.video not in meta_full:
        print(f"ERROR: video '{args_cli.video}' not in {MEVIS_META}")
        sys.exit(1)
    expressions = meta_full[args_cli.video]["expressions"]
    if args_cli.exp not in expressions:
        print(f"ERROR: exp '{args_cli.exp}' not in video '{args_cli.video}'")
        print(f"available exp ids: {list(expressions.keys())[:10]}{'...' if len(expressions)>10 else ''}")
        sys.exit(1)
    exp_text = expressions[args_cli.exp]["exp"]
    frames = meta_full[args_cli.video]["frames"]
    print(f"  exp text   : {exp_text!r}")
    print(f"  num frames : {len(frames)}")

    # ---- load CLIP scorer (used by Stage 1 + Stage 3) ---------------------
    print("\n[setup] loading CLIPScorer ...")
    scorer = ImageScorer()

    # ---- vLLM client + call counter ---------------------------------------
    print("\n[setup] connecting to vLLM ...")
    vl_model = load_vlm()
    counter = CallCounter(vl_model["client"])

    # ---- Stage 1: CLIP scoring (per-sample, no amortization) --------------
    print("\n[stage 1] CLIP scoring (per-sample) ...")
    t0 = time.perf_counter()
    clip_scores = stage1_clip_score_per_sample(scorer, args_cli.video, frames, exp_text)
    t_stage1 = time.perf_counter() - t0
    print(f"  → {t_stage1:.2f} sec  ({len(frames)} image encodes + 1 text encode)")

    # ---- Stage 2: MLLM preprocess (convert_query + generate_qa) -----------
    print("\n[stage 2] MLLM preprocess (vLLM) ...")
    counter.reset()
    pre_args = Args(max_query_num=args_cli.max_query_num)
    t0 = time.perf_counter()
    description, type_ = convert_query_with_vlm(exp_text, vl_model)
    if not exp_text.strip().endswith("?"):
        description = exp_text.strip()
    else:
        description = (description or "").strip()
    if type_ == 3:
        qa_complete = generate_qa_with_vlm(description, type_, vl_model)
    else:
        qa_complete = []
    t_stage2 = time.perf_counter() - t0
    print(f"  → {t_stage2:.2f} sec  ({counter.calls} vLLM calls, "
          f"{counter.prompt_tokens} prompt + {counter.completion_tokens} gen tokens)")
    print(f"    description: {description!r}")
    print(f"    type       : {type_}")
    print(f"    qa pairs   : {len(qa_complete)}")

    # ---- Stage 3: Agent inference (process_one_sample) --------------------
    print("\n[stage 3] Agent inference (vLLM + CLIP runtime) ...")
    counter.reset()
    # build minimal lookups process_one_sample expects
    all_query_qa = {f"{args_cli.video}/{args_cli.exp}": {
        "exp": exp_text,
        "description": description,
        "type": type_,
        "qa_complete": qa_complete,
        "static": qa_complete,  # process_one_sample reads .get('static', ...)
    }}
    all_CLIP_query_scores = {args_cli.video: {"CLIP": {args_cli.exp: clip_scores}}}

    # preload images (the production sub_processor does this once per video)
    src_imgs, bytes_imgs = [], []
    for fname in frames:
        img = Image.open(os.path.join(MEVIS_IMG_FOLDER, args_cli.video, fname + ".jpg")).convert("RGB")
        src_imgs.append(img)
        bytes_imgs.append(np.asarray(img))

    agent_args = Args(
        max_query_num=args_cli.max_query_num,
        max_update_frame_num=args_cli.max_update_frame_num,
        max_update_descriptions_num=args_cli.max_update_descriptions_num,
    )

    # output dir for the agent's internal save_path / feedback file
    save_path_prefix = "/tmp/refer_agent_measure"
    os.makedirs(save_path_prefix, exist_ok=True)

    t0 = time.perf_counter()
    entry = process_one_sample(
        video=args_cli.video,
        exp=exp_text,
        exp_id=args_cli.exp,
        frames=frames,
        src_imgs=src_imgs,
        bytes_imgs=bytes_imgs,
        img_folder=MEVIS_IMG_FOLDER,
        save_path_prefix=save_path_prefix,
        split_frames_idx=0,
        args=agent_args,
        vl_model=vl_model,
        scorer=scorer,
        all_query_qa=all_query_qa,
        all_CLIP_query_scores=all_CLIP_query_scores,
        pid=0,
    )
    t_stage3 = time.perf_counter() - t0
    print(f"  → {t_stage3:.2f} sec  ({counter.calls} vLLM calls, "
          f"{counter.prompt_tokens} prompt + {counter.completion_tokens} gen tokens)")
    print(f"    boxes      : {len(entry['all_boxes'])}")
    print(f"    descriptions: {entry['descriptions']}")
    print(f"    keyframe   : {entry['key_frame_idx']}")

    # ---- Stage 4: SAM2 mask propagation ----------------------------------
    if args_cli.skip_stage4:
        print("\n[stage 4] SKIPPED (--skip-stage4)")
        t_stage4 = 0.0
    else:
        print("\n[stage 4] SAM2 mask propagation ...")
        t0 = time.perf_counter()
        sam2_info = stage4_sam2(entry)
        t_stage4 = time.perf_counter() - t0
        print(f"  → {t_stage4:.2f} sec  ({sam2_info})")

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Per-sample wall-clock summary")
    print("=" * 70)
    print(f"  Stage 1  CLIP scoring          : {t_stage1:8.2f} sec")
    print(f"  Stage 2  MLLM preprocess       : {t_stage2:8.2f} sec")
    print(f"  Stage 3  Agent inference       : {t_stage3:8.2f} sec")
    print(f"  Stage 4  SAM2 propagation      : {t_stage4:8.2f} sec"
          + ("  (skipped)" if args_cli.skip_stage4 else ""))
    print(f"  ─────────────────────────────────────────────────")
    total = t_stage1 + t_stage2 + t_stage3 + t_stage4
    print(f"  Total                          : {total:8.2f} sec")


if __name__ == "__main__":
    main()
