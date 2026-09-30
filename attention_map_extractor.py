"""
Attention Map Extractor — Fixed Weight Loading
Run:
  uv run python attention_map_extractor.py \
    --args.json-baseline results/tables/trajectories_shift_0cm_*.json \
    --args.json-ood15    results/tables/trajectories_shift_15cm_*.json \
    --args.json-ood25    results/tables/trajectories_shift_25cm_*.json \
    --args.timestep 100
"""

import dataclasses
import json
import logging
import pathlib
import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import tyro
from PIL import Image as PILImage

_HERE     = pathlib.Path(__file__).parent
PLOTS_DIR = _HERE / "results" / "plots" / "attention"
LOGS_DIR  = _HERE / "logs"

WEIGHTS_PATH = pathlib.Path(
    "/mnt/home/mubashar/.cache/openpi/openpi-assets/"
    "checkpoints/pi05_libero_pytorch/model.safetensors"
)

# CORRECT prefix found from debug output
VISION_PREFIX = (
    "paligemma_with_expert.paligemma.model."
    "vision_tower.vision_model."
)


@dataclasses.dataclass
class Args:
    json_baseline: str = ""
    json_ood15:    str = ""
    json_ood25:    str = ""
    timestep:      int = 100
    episode:       int = 0
    layer:         int = -1


def load_siglip():
    from transformers import SiglipVisionModel, SiglipVisionConfig
    from safetensors.torch import load_file

    logging.info(f"Loading weights: {WEIGHTS_PATH}")
    sd = load_file(str(WEIGHTS_PATH))

    # Extract vision keys using correct prefix
    vision_sd = {}
    for k, v in sd.items():
        if k.startswith(VISION_PREFIX):
            # Model expects "vision_model.X", keys have just "X"
            new_key = "vision_model." + k[len(VISION_PREFIX):]
            vision_sd[new_key] = v

    logging.info(f"Vision keys found: {len(vision_sd)}")
    assert len(vision_sd) > 100, \
        f"Too few keys ({len(vision_sd)}) — prefix wrong"

    # Sample keys
    for k in list(vision_sd.keys())[:5]:
        logging.info(f"  {k}: {vision_sd[k].shape}")

    # Build model
    config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        attn_implementation="eager",
    )
    model = SiglipVisionModel(config)

    missing, unexpected = model.load_state_dict(vision_sd, strict=False)
    logging.info(f"Missing={len(missing)}, Unexpected={len(unexpected)}")
    if missing:
        logging.info(f"  Sample missing: {missing[:3]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()
    logging.info(f"Model ready on {device}")
    return model, device


def preprocess(img_np, device):
    img = img_np[:, :, :3].astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t   = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return t.to(device, dtype=torch.float32)


def get_heatmap(model, img_tensor, layer_idx=-1):
    with torch.no_grad():
        out = model(pixel_values=img_tensor, output_attentions=True)

    if not out.attentions:
        return None

    # [1, heads, seq, seq]
    attn = out.attentions[layer_idx].squeeze(0).cpu().float().numpy()
    n_heads, seq, _ = attn.shape

    # Average over heads → [seq, seq]
    avg = attn.mean(0)

    # seq = 1 (CLS) + n_patches
    cls_attn = avg[0, 1:]          # CLS → patches
    n_p      = len(cls_attn)
    n_side   = int(np.sqrt(n_p))
    n_p_sq   = n_side * n_side
    cls_attn = cls_attn[:n_p_sq]   # trim to perfect square

    mn, mx = cls_attn.min(), cls_attn.max()
    if mx > mn:
        cls_attn = (cls_attn - mn) / (mx - mn)

    grid = cls_attn.reshape(n_side, n_side)
    heat = np.array(
        PILImage.fromarray((grid * 255).astype(np.uint8))
        .resize((224, 224), PILImage.BILINEAR)
    ) / 255.0
    return heat


def load_frame(vid_dir, episode, timestep, wait=10):
    ep   = episode + 1
    vids = list(pathlib.Path(vid_dir).glob(f"ep{ep:02d}_*.mp4"))
    if not vids:
        logging.warning(f"No video: {vid_dir}")
        return None
    reader = imageio.get_reader(str(vids[0]))
    frames = [f for f in reader]
    reader.close()
    idx = min(max(0, timestep - wait), len(frames) - 1)
    logging.info(f"  {vids[0].name}: frame {idx}/{len(frames)}")
    return frames[idx]


def run_analysis(args: Args) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"attn_{ts}.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 60)
    logging.info(f"ATTENTION EXTRACTOR  t={args.timestep}")
    logging.info("=" * 60)

    # Load model with real weights
    model, device = load_siglip()

    # Video dirs
    video_dirs = {
        "Baseline (0cm)": _HERE / "results/videos/traj_shift_0cm",
        "OOD +15cm":      _HERE / "results/videos/traj_shift_15cm",
        "OOD +25cm":      _HERE / "results/videos/traj_shift_25cm",
    }

    cfgs = [
        ("Baseline (0cm)", args.json_baseline, "blue"),
        ("OOD +15cm",      args.json_ood15,    "red"),
        ("OOD +25cm",      args.json_ood25,    "orange"),
    ]
    cfgs = [(l, j, c) for l, j, c in cfgs if j]

    # Extract frames + attention
    results = {}
    for label, _, color in cfgs:
        logging.info(f"\n{label}")
        frame = load_frame(video_dirs[label],
                           args.episode, args.timestep)
        if frame is None:
            continue
        img_t = preprocess(frame, device)
        heat  = get_heatmap(model, img_t, args.layer)
        if heat is not None:
            results[label] = {
                "frame": frame[:, :, :3],
                "heat":  heat,
                "color": color,
            }
            logging.info(f"  heat min={heat.min():.3f} max={heat.max():.3f}")

    if not results:
        logging.error("No results")
        return

    # ── Main comparison figure ─────────────────────────────────
    n   = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (label, res) in enumerate(results.items()):
        frame = res["frame"]
        heat  = res["heat"]

        axes[0, col].imshow(frame)
        axes[0, col].set_title(
            f"{label}\nt={args.timestep}",
            fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(frame)
        im = axes[1, col].imshow(
            heat, alpha=0.55, cmap='jet', vmin=0, vmax=1)
        axes[1, col].set_title(
            "SigLIP Attention\n(RED = model focuses here)",
            fontsize=11)
        axes[1, col].axis('off')

    plt.colorbar(im, ax=axes[1, :], shrink=0.6,
                 label='Attention intensity')
    fig.suptitle(
        f"SigLIP Attention Maps — Memory Trap Root Cause\n"
        f"t={args.timestep} | Real π₀.₅ weights loaded",
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = PLOTS_DIR / f"attn_compare_t{args.timestep}_{ts}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"\nComparison saved: {out}")

    # ── Over-time figure ───────────────────────────────────────
    timesteps = [50, 100, 150, 200, 300]
    n_t = len(timesteps)
    n_l = len(results)
    fig2, axes2 = plt.subplots(n_l, n_t, figsize=(4*n_t, 4*n_l))
    if n_l == 1:
        axes2 = axes2.reshape(1, -1)

    row_labels = list(results.keys())
    for row, label in enumerate(row_labels):
        for col, t_val in enumerate(timesteps):
            frame = load_frame(video_dirs[label],
                               args.episode, t_val)
            ax = axes2[row, col]
            if frame is not None:
                img_t = preprocess(frame, device)
                heat  = get_heatmap(model, img_t, args.layer)
                ax.imshow(frame[:, :, :3])
                if heat is not None:
                    ax.imshow(heat, alpha=0.55, cmap='jet',
                              vmin=0, vmax=1)
            ax.set_title(f"t={t_val}", fontsize=9)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=9,
                              rotation=0, labelpad=70,
                              va='center')

    fig2.suptitle(
        "Attention Over Time — Real π₀.₅ Weights\n"
        "Row 1=Baseline, Row 2=+15cm, Row 3=+25cm",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = PLOTS_DIR / f"attn_overtime_{ts}.png"
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Over-time: {out2}")

    logging.info("\nDONE — " + str(PLOTS_DIR))


if __name__ == "__main__":
    tyro.cli(run_analysis)