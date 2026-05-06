#!/usr/bin/env python3
"""Plot extracted ETPI hidden-state trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords.astype(np.float32)


def tsne_2d(matrix: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    n_samples = matrix.shape[0]
    if n_samples < 3:
        return pca_2d(matrix)
    safe_perplexity = min(perplexity, max(1.0, (n_samples - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="cosine",
    )
    return tsne.fit_transform(matrix).astype(np.float32)


def segment_colors(segment: str) -> str:
    if segment == "prefill":
        return "#111827"
    if segment == "think_end":
        return "#dc2626"
    return "#2563eb"


def plot_coords(
    coords: np.ndarray,
    rows: list[dict[str, str]],
    *,
    title: str,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    colors = [segment_colors(row.get("segment", "")) for row in rows]
    ax.plot(coords[:, 0], coords[:, 1], color="#9ca3af", linewidth=1.0, alpha=0.65, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.9, zorder=2)

    for index in [0, len(rows) - 1]:
        ax.scatter(coords[index, 0], coords[index, 1], c=colors[index], s=90, edgecolors="white", linewidths=1.5, zorder=3)
        label = "prefill" if index == 0 else rows[index].get("token_text") or "end"
        ax.annotate(label, (coords[index, 0], coords[index, 1]), xytext=(6, 6), textcoords="offset points")

    end_indices = [i for i, row in enumerate(rows) if row.get("segment") == "think_end"]
    for index in end_indices:
        ax.scatter(coords[index, 0], coords[index, 1], c=colors[index], s=60, marker="s", edgecolors="white", linewidths=1.0, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA or t-SNE views for an extracted hidden-state trajectory.")
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--method", choices=["pca", "tsne"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latent_dir = args.latent_dir
    hidden_path = latent_dir / "hidden_states.npz"
    distances_path = latent_dir / f"distances_layer_{args.layer}.csv"
    if not hidden_path.exists():
        raise FileNotFoundError(hidden_path)
    if not distances_path.exists():
        raise FileNotFoundError(distances_path)

    hidden = np.load(hidden_path)
    key = f"layer_{args.layer}"
    if key not in hidden:
        raise KeyError(f"{key} not in {hidden_path}; available: {sorted(hidden.files)}")
    matrix = hidden[key]
    rows = read_csv(distances_path)
    if len(rows) != matrix.shape[0]:
        raise ValueError(f"{distances_path} has {len(rows)} rows but {key} has {matrix.shape[0]} vectors")

    if args.method == "pca":
        coords = pca_2d(matrix)
    else:
        coords = tsne_2d(matrix, perplexity=args.perplexity, seed=args.seed)

    prefix = args.out_prefix or f"{args.method}_layer_{args.layer}"
    csv_path = latent_dir / f"{prefix}.csv"
    png_path = latent_dir / f"{prefix}.png"
    out_rows = [
        {**row, f"{args.method}_x": float(coord[0]), f"{args.method}_y": float(coord[1])}
        for row, coord in zip(rows, coords)
    ]
    write_csv(csv_path, out_rows)

    metadata_path = latent_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    title = f"{args.method.upper()} hidden trajectory layer {args.layer}"
    if metadata.get("step_id"):
        title += f"\n{metadata['step_id']}"
    plot_coords(coords, rows, title=title, out_png=png_path)
    print(json.dumps({"csv": str(csv_path), "png": str(png_path), "points": int(matrix.shape[0])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
