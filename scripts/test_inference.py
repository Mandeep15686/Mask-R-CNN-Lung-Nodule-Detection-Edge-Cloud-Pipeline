# scripts/test_inference.py  — FULL REPLACEMENT
import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Settings ──────────────────────────────────────────────────
IMAGE_PATH      = "./data/sample_patches/ct.png"
CHECKPOINT_PATH = "./experiments/G-all5-ep10/best_model.pth"
SCORE_THRESHOLD = 0.15   # lowered from 0.3 — catch more candidates
PATCH_SIZE      = 64
STRIDE          = 32     # overlap between windows (50% overlap)

# ── Load model ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from training.model import LungNoduleMaskRCNN

ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
cfg  = ckpt.get("config", {})
mask_loss_cfg = cfg.get("mask_loss",
    {"alpha": 0.3, "beta": 0.3, "gamma": 0.5, "delta": 0.1, "epsilon": 0.1})

model = LungNoduleMaskRCNN(num_classes=2, mask_loss_cfg=mask_loss_cfg)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.to(device)
model.eval()
print(f"Model loaded: {CHECKPOINT_PATH}")

# ── Load full CT image ────────────────────────────────────────
original = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = original.size
print(f"Full image size: {orig_w} × {orig_h} px")

# Resize to a workable size (keep aspect ratio)
WORK_SIZE = 512
scale = WORK_SIZE / max(orig_w, orig_h)
work_w = int(orig_w * scale)
work_h = int(orig_h * scale)
work_img = original.resize((work_w, work_h), Image.LANCZOS)
work_arr = np.array(work_img, dtype=np.float32) / 255.0  # (H, W, 3)
print(f"Working size: {work_w} × {work_h} px")

# ── Sliding window inference ───────────────────────────────────
print(f"\nRunning sliding window (patch={PATCH_SIZE}, stride={STRIDE})...")

all_detections = []   # each = (x1, y1, x2, y2, score, mask_patch, px_offset)

rows = range(0, work_h - PATCH_SIZE + 1, STRIDE)
cols = range(0, work_w - PATCH_SIZE + 1, STRIDE)
total_patches = len(rows) * len(cols)
print(f"Total patches to scan: {total_patches}")

with torch.no_grad():
    for patch_idx, row in enumerate(rows):
        for col in cols:
            # Extract patch
            patch = work_arr[row:row+PATCH_SIZE, col:col+PATCH_SIZE]  # (64,64,3)
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue

            patch_t = torch.tensor(patch.transpose(2, 0, 1))  # (3,64,64)

            # Run model
            out = model([patch_t.to(device)])[0]

            scores_p = out.get("scores", torch.zeros(0)).cpu().numpy()
            boxes_p  = out.get("boxes",  torch.zeros(0, 4)).cpu().numpy()
            masks_p  = out.get("masks",  None)

            # Keep detections above threshold
            for i, score in enumerate(scores_p):
                if score >= SCORE_THRESHOLD:
                    bx1, by1, bx2, by2 = boxes_p[i]
                    # Map back to working-image coordinates
                    abs_x1 = col + bx1
                    abs_y1 = row + by1
                    abs_x2 = col + bx2
                    abs_y2 = row + by2
                    mask_arr = None
                    if masks_p is not None and i < len(masks_p):
                        mask_arr = masks_p[i, 0].cpu().numpy()

                    all_detections.append({
                        "x1": abs_x1, "y1": abs_y1,
                        "x2": abs_x2, "y2": abs_y2,
                        "score": float(score),
                        "mask": mask_arr,
                        "patch_col": col, "patch_row": row,
                    })

print(f"Raw detections before NMS: {len(all_detections)}")

# ── Non-Maximum Suppression (merge overlapping boxes) ──────────
def iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
    ab = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])
    return inter / (aa + ab - inter + 1e-6)

def nms(dets, iou_thresh=0.3):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best, d) < iou_thresh]
    return kept

final_dets = nms(all_detections, iou_thresh=0.3)
print(f"After NMS: {len(final_dets)} detections")

# ── Print results ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  DETECTED NODULES: {len(final_dets)}")
print(f"{'='*55}")

for i, d in enumerate(final_dets):
    # Map back to original image coordinates
    orig_x1 = d["x1"] / scale
    orig_y1 = d["y1"] / scale
    orig_x2 = d["x2"] / scale
    orig_y2 = d["y2"] / scale
    w_px = orig_x2 - orig_x1
    h_px = orig_y2 - orig_y1
    diam_mm = ((w_px + h_px) / 2) * 0.7
    print(f"  Nodule {i+1}:")
    print(f"    Confidence : {d['score']:.4f}  ({d['score']*100:.1f}%)")
    print(f"    Location   : x={orig_x1:.0f}–{orig_x2:.0f}, y={orig_y1:.0f}–{orig_y2:.0f} px")
    print(f"    Size       : {w_px:.0f} × {h_px:.0f} px")
    print(f"    Est. diam  : {diam_mm:.1f} mm")
    if diam_mm < 6:
        print(f"    Category   : Small nodule (<6mm)")
    elif diam_mm < 20:
        print(f"    Category   : Medium nodule (6–20mm)")
    else:
        print(f"    Category   : Large mass (>20mm)")

if not final_dets:
    print("  No nodules detected.")
    print("  Try lowering SCORE_THRESHOLD to 0.05")

# ── Visualise ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(f"Mask R-CNN — {os.path.basename(IMAGE_PATH)}", fontsize=13)

# Panel 1: Detections on working image
axes[0].imshow(work_img)
axes[0].set_title(f"Detections: {len(final_dets)} nodule(s)  [threshold={SCORE_THRESHOLD}]")
for d in final_dets:
    rect = patches.Rectangle(
        (d["x1"], d["y1"]), d["x2"]-d["x1"], d["y2"]-d["y1"],
        linewidth=2, edgecolor="cyan", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].text(
        d["x1"], d["y1"] - 3,
        f"{d['score']:.2f}",
        color="cyan", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7)
    )
axes[0].axis("off")

# Panel 2: Heatmap of all detection scores across full image
heatmap = np.zeros((work_h, work_w), dtype=np.float32)
for d in all_detections:   # use ALL (before NMS) for heatmap
    x1 = int(max(0, d["x1"])); y1 = int(max(0, d["y1"]))
    x2 = int(min(work_w, d["x2"])); y2 = int(min(work_h, d["y2"]))
    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], d["score"])

axes[1].imshow(work_img, alpha=0.6)
if heatmap.max() > 0:
    axes[1].imshow(heatmap, alpha=0.5, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="hot"),
        ax=axes[1], shrink=0.6, label="Detection Score"
    )
axes[1].set_title("Score Heatmap (all candidates)")
axes[1].axis("off")

plt.tight_layout()
out_path = "./output_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
