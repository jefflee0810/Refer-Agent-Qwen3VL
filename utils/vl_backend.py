"""VLM backend for Refer-Agent.

Talks to a vLLM OpenAI-compatible server (e.g. started via
`scripts/serve_vllm.sh <gpu_id> <tp_size>`).

Public API kept stable so existing helpers in eval/utils don't change:
    load_vlm(device) -> handle (device is ignored, accepted for backward compat)
    vl_chat(handle, messages, **kw) -> str
    parse_grounding_bbox(response, image_size=None) -> Optional[List[float]]
    parse_point(response, image_size=None) -> Optional[List[float]]

Configuration (env vars):
    REFER_VLLM_BASE_URL  default http://localhost:10000/v1
    REFER_VLLM_MODEL     default qwen3-vl-8b-thinking   (= --served-model-name)
    REFER_VLLM_API_KEY   default "EMPTY"
    REFER_VLLM_TIMEOUT   default 3600
"""
import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

VLLM_BASE_URL = os.environ.get("REFER_VLLM_BASE_URL", "http://localhost:10000/v1")
VLLM_MODEL = os.environ.get("REFER_VLLM_MODEL", "qwen3-vl-8b-thinking")
VLLM_API_KEY = os.environ.get("REFER_VLLM_API_KEY", "EMPTY")
VLLM_TIMEOUT = float(os.environ.get("REFER_VLLM_TIMEOUT", "3600"))


# ---------------------------------------------------------------------------
# Backend handle / chat
# ---------------------------------------------------------------------------

def load_vlm(device: Any = None) -> Dict[str, Any]:
    """Create an OpenAI client connected to the vLLM server.

    `device` is ignored (kept for backward-compat with the in-process backend
    signature). Each subprocess in mp.Process should call this independently
    to get its own client; the vLLM server handles concurrency.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        timeout=VLLM_TIMEOUT,
    )
    return {
        "backend": "vllm",
        "client": client,
        "model": VLLM_MODEL,
    }


_WEBP_MAX_DIM = 16383  # WebP encoder's per-side pixel limit.


def _pil_to_data_uri(img: Image.Image) -> str:
    """Encode PIL image as a base64 WEBP-lossless data URI (matches seungho/src style).

    If any side exceeds WebP's per-side 16383 pixel limit (e.g. multi-frame
    stitched images), downsize to fit while preserving aspect ratio.
    """
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(img)}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w > _WEBP_MAX_DIM or h > _WEBP_MAX_DIM:
        ratio = min(_WEBP_MAX_DIM / w, _WEBP_MAX_DIM / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/webp;base64,{b64}"


def _to_openai_content(content: Any) -> Any:
    """Convert Ovis-flavored content blocks to OpenAI chat format.

    Input items:
        {"type": "image", "image": <PIL.Image>}    -> {"type": "image_url", "image_url": {"url": data_uri}}
        {"type": "text",  "text":  "..."}          -> kept as-is
    Strings are returned untouched.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    out: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            out.append(item)
            continue
        t = item.get("type")
        if t == "image":
            uri = _pil_to_data_uri(item["image"])
            out.append({"type": "image_url", "image_url": {"url": uri}})
        elif t == "image_url":
            out.append(item)
        elif t == "text":
            out.append({"type": "text", "text": item["text"]})
        else:
            out.append(item)
    return out


def vl_chat(
    handle: Dict[str, Any],
    messages: List[Dict[str, Any]],
    *,
    thinking: bool = True,
    max_pixels: Optional[int] = None,  # ignored on vLLM path (configured server-side)
    max_new_tokens: int = 4096,
    thinking_budget: int = 3584,        # ignored on vLLM path
    do_sample: bool = True,
    temperature: Optional[float] = None,
) -> str:
    """Send a chat completion request to the vLLM server, return decoded text.

    Returns the raw assistant message content; thinking models include
    `<think>...</think>` blocks which existing parsers strip via split.
    """
    client = handle["client"]
    model = handle["model"]

    payload_messages = []
    for m in messages:
        payload_messages.append({
            "role": m.get("role", "user"),
            "content": _to_openai_content(m.get("content")),
        })

    if temperature is None:
        temperature = 0.7 if do_sample else 0.0

    completion = client.chat.completions.create(
        model=model,
        messages=payload_messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Grounding parsers
# ---------------------------------------------------------------------------

_BOX_TAG_RE = re.compile(
    r'<box>\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\},\s]*\s*'
    r'[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]*\s*</box>'
)
_BBOX2D_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]'
)
_COORD_FALLBACKS = [
    re.compile(
        r'[\(\[\{]\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}][,\s]*'
        r'[\(\[\{]\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]'
    ),
    re.compile(r'(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)'),
    re.compile(r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'),
    re.compile(r'\b(\d{2,4})\b\s+\b(\d{2,4})\b\s+\b(\d{2,4})\b\s+\b(\d{2,4})\b'),
]
_POINT_TAG_RE = re.compile(
    r'<point>\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]*\s*</point>'
)


def _strip_thinking(response: str) -> str:
    response = response.strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    response = response.replace("<|box_start|>", "<box>").replace("<|box_end|>", "</box>")
    response = response.replace("<|point_start|>", "<point>").replace("<|point_end|>", "</point>")
    response = response.replace("<bbox>", "<box>").replace("</bbox>", "</box>")
    response = response.replace("<points>", "<point>").replace("</points>", "</point>")
    return response


def _normalize_xyxy(x1, y1, x2, y2, image_size: Optional[Tuple[int, int]]) -> List[float]:
    """Normalize raw coords from a Qwen-style VLM response to [0,1] xyxy.

    Heuristic:
      - max coord <= 1.0  -> already normalized.
      - image_size given  -> Qwen3-VL emits absolute pixel coords; divide by (w,h).
      - else fallback     -> assume 0..1000 normalization (Ovis convention).
    """
    coords = [float(x1), float(y1), float(x2), float(y2)]
    max_v = max(coords)
    if max_v <= 1.0:
        norm = coords
    elif image_size is not None:
        w, h = image_size
        norm = [coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h]
    elif max_v <= 1000.0:
        norm = [c / 1000.0 for c in coords]
    else:
        norm = [c / 1000.0 for c in coords]
    return [max(0.0, min(1.0, v)) for v in norm]


def _normalize_xy(x, y, image_size: Optional[Tuple[int, int]]) -> List[float]:
    coords = [float(x), float(y)]
    max_v = max(coords)
    if max_v <= 1.0:
        norm = coords
    elif image_size is not None:
        w, h = image_size
        norm = [coords[0] / w, coords[1] / h]
    elif max_v <= 1000.0:
        norm = [c / 1000.0 for c in coords]
    else:
        norm = [c / 1000.0 for c in coords]
    return [max(0.0, min(1.0, v)) for v in norm]


def _dedup(boxes: List[List[float]]) -> List[List[float]]:
    seen = set()
    uniq: List[List[float]] = []
    for b in boxes:
        key = tuple(round(v, 3) for v in b)
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq


def parse_grounding_bbox(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[List[float]]:
    text = _strip_thinking(response)
    boxes: List[List[float]] = []

    for m in _BOX_TAG_RE.findall(text):
        try:
            boxes.append(_normalize_xyxy(*m, image_size=image_size))
        except (ValueError, TypeError):
            pass

    for m in _BBOX2D_RE.findall(text):
        try:
            boxes.append(_normalize_xyxy(*m, image_size=image_size))
        except (ValueError, TypeError):
            pass

    if not boxes:
        for pat in _COORD_FALLBACKS:
            for m in pat.findall(text):
                try:
                    if len(m) == 4:
                        boxes.append(_normalize_xyxy(*m, image_size=image_size))
                except (ValueError, TypeError):
                    continue

    boxes = _dedup(boxes)
    if not boxes:
        return None
    bx = boxes[-1]
    if bx[2] < bx[0]:
        bx[0], bx[2] = bx[2], bx[0]
    if bx[3] < bx[1]:
        bx[1], bx[3] = bx[3], bx[1]
    return bx


def parse_point(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[List[float]]:
    text = _strip_thinking(response)
    points: List[List[float]] = []
    for m in _POINT_TAG_RE.findall(text):
        try:
            points.append(_normalize_xy(*m, image_size=image_size))
        except (ValueError, TypeError):
            continue
    if not points:
        for m in re.findall(r'"point_2d"\s*:\s*\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]', text):
            try:
                points.append(_normalize_xy(*m, image_size=image_size))
            except (ValueError, TypeError):
                continue
    if not points:
        return None
    return points[-1]
