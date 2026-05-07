"""Smoke test for the vLLM backend.

Prerequisite: vLLM OpenAI server is running. Start it in a separate terminal:
    bash scripts/serve_vllm.sh <gpu_id> <tp_size>     # e.g. 4 1

Then in another terminal:
    PYTHONPATH=. python scripts/sanity_qwen.py [image_path] [query]

Defaults to assets/refer-agent.jpg with a generic grounding query. Prints the
raw model response and the parsed [0,1] xyxy bbox.
"""
import argparse
import os
import sys

from PIL import Image

from utils.vl_backend import (
    VLLM_BASE_URL,
    VLLM_MODEL,
    load_vlm,
    vl_chat,
    parse_grounding_bbox,
)


GROUNDING_PROMPT = """# Task
Identify the unique target object in the image that corresponds to the description and provide the bounding box coordinates for that object.

# Input
description: {description}
An original query: {query}

# Instruction
1. Find the <ref>{description}</ref> in the image. Compare the difference between objects and find the **most closely matched one**.
2. The description corresponds to the original query "{query}" in this image, so you can also obtain information about the target from the query if necessary.
3. If you find upon careful consideration that there is no corresponding object in the image, return empty.

Please give the coordinates of the bounding box.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="assets/refer-agent.jpg")
    ap.add_argument("query", nargs="?", default="the main subject in the image")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    description = args.description or args.query

    print(f"[sanity] vLLM base url = {VLLM_BASE_URL}")
    print(f"[sanity] vLLM model    = {VLLM_MODEL}")
    print(f"[sanity] image         = {args.image}")
    print(f"[sanity] query         = {args.query}")
    if not os.path.exists(args.image):
        print(f"[sanity] ERROR: image not found: {args.image}")
        sys.exit(1)

    handle = load_vlm()
    print(f"[sanity] client created, backend = {handle['backend']}")

    image = Image.open(args.image).convert("RGB")
    print(f"[sanity] image size = {image.size}")

    messages = [
        {"role": "system", "content": "You are a helpful image grounding assistant"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": GROUNDING_PROMPT.format(query=args.query, description=description)},
            ],
        },
    ]

    response = vl_chat(
        handle, messages,
        thinking=False, max_pixels=896 * 896,
        max_new_tokens=2048, thinking_budget=1024,
    )
    print("\n===== RAW RESPONSE =====")
    print(response)
    print("========================\n")

    bbox = parse_grounding_bbox(response, image_size=image.size)
    print(f"[sanity] parsed bbox (normalized [0,1] xyxy): {bbox}")
    if bbox is not None:
        w, h = image.size
        px = [int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)]
        print(f"[sanity] pixel bbox: {px}")
        print("[sanity] OK")
    else:
        print("[sanity] WARNING: no bbox parsed; check raw response and parser regex.")


if __name__ == "__main__":
    main()
