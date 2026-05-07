#!/usr/bin/env python3
import json
import os
import argparse
import warnings
from typing import Dict, List, Optional

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings("ignore")


class ImageScorer:
    """CLIP-based scorer with per-video batched image/text encoding."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        progress_desc: Optional[str] = None,
    ) -> torch.Tensor:
        feats = []
        idx_iter = range(0, len(image_paths), batch_size)
        if progress_desc is not None:
            idx_iter = tqdm(idx_iter, desc=progress_desc, leave=False)
        for i in idx_iter:
            batch = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
            inputs = self.clip_processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            vision_out = self.clip_model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output
            f = self.clip_model.visual_projection(pooled)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        text_out = self.clip_model.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = text_out.pooler_output
        t = self.clip_model.text_projection(pooled)
        t = t / t.norm(dim=-1, keepdim=True)
        return t

    @torch.no_grad()
    def calculate_clip_score(self, image_path: str, text: str) -> float:
        """Single-pair convenience wrapper. Returns cosine similarity."""
        if not text.strip():
            return 0.0
        img_feat = self.encode_images([image_path])
        txt_feat = self.encode_texts([text])
        return float((img_feat @ txt_feat.T).item())

    @staticmethod
    def normalize_scores(scores_dict: Dict[str, float], invert: bool = False) -> Dict[str, float]:
        if not scores_dict:
            return {}
        scores = list(scores_dict.values())
        mean = sum(scores) / len(scores)
        std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        if std == 0:
            return {k: 5.5 for k in scores_dict.keys()}
        out = {}
        for k, s in scores_dict.items():
            z = (s - mean) / std
            if invert:
                z = -z
            out[k] = max(1.0, min(10.0, 5.5 + 1.5 * z))
        return out


def batch_score_videos(base_dir: str) -> Dict[str, Dict]:
    meta = json.load(
        open("/home/cvlab18/media/data2/datasets/revos/meta_expressions_valid_.json")
    )["videos"]

    scorer = ImageScorer()
    video_scores = {}

    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    for video_dir_name in tqdm(meta.keys(), desc="videos"):
        video_path = os.path.join(base_dir, video_dir_name)

        image_files = sorted(
            [f for f in os.listdir(video_path) if f.lower().endswith(supported_formats)],
            key=lambda x: int("".join(filter(str.isdigit, x))),
        )
        if not image_files:
            print(f"No supported image files found in '{video_dir_name}'.")
            continue

        image_paths = [os.path.join(video_path, f) for f in image_files]
        image_feats = scorer.encode_images(
            image_paths, progress_desc=f"images {video_dir_name}"
        )

        exp_ids = list(meta[video_dir_name]["expressions"].keys())
        texts = [meta[video_dir_name]["expressions"][e]["exp"] for e in exp_ids]
        text_feats = scorer.encode_texts(texts)
        sim_matrix = (image_feats @ text_feats.T).cpu().numpy()

        video_scores[video_dir_name] = {"CLIP": {}}
        for j, exp_id in enumerate(
            tqdm(exp_ids, desc=f"CLIP exps {video_dir_name}", leave=False)
        ):
            raw = {image_files[i]: float(sim_matrix[i, j]) for i in range(len(image_files))}
            normalized = scorer.normalize_scores(raw)
            video_scores[video_dir_name]["CLIP"][exp_id] = {
                f: {"clip": round(normalized[f], 2),
                    "weighted_score": round(normalized[f], 2)}
                for f in image_files
            }

    return video_scores


def main():
    parser = argparse.ArgumentParser(description="CLIP frame–expression scoring (ReVOS).")
    parser.add_argument(
        "--input_dir",
        default="/home/cvlab18/media/data2/datasets/revos/JPEGImages",
        help="Directory containing per-video subdirectories of frame images.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output_concat/revos/CLIP/video_scores_revos.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found - {args.input_dir}")
        return

    print(f"Starting batch CLIP scoring of '{args.input_dir}'...")
    all_video_scores = batch_score_videos(args.input_dir)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_video_scores, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()
