"""
Plot KITTI trajectory comparison across multiple methods.
Reads GT poses (KITTI format) and estimated poses (TUM format),
aligns via Sim(3), and generates side-by-side comparison plots.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def read_kitti_poses(filepath):
    """Read KITTI ground-truth pose file (3x4 matrix per line)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return poses


def read_tum_poses(filepath):
    """Read TUM-format trajectory (timestamp tx ty tz qx qy qz qw)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            if len(vals) >= 8:
                _, tx, ty, tz, qx, qy, qz, qw = vals[:8]
                T = np.eye(4)
                T[:3, 3] = [tx, ty, tz]
                T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
                poses.append(T)
    return poses


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def align_sim3(gt_xyz, es_xyz):
    """Sim(3) Umeyama alignment of estimated trajectory to GT."""
    n = gt_xyz.shape[0]
    mu_gt = gt_xyz.mean(axis=0)
    mu_es = es_xyz.mean(axis=0)
    gt_c = gt_xyz - mu_gt
    es_c = es_xyz - mu_es

    sigma_gt = np.sum(gt_c ** 2) / n
    sigma_es = np.sum(es_c ** 2) / n
    cov = (gt_c.T @ es_c) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / sigma_es
    t = mu_gt - scale * R @ mu_es

    aligned = scale * (es_xyz @ R.T) + t
    return aligned, scale, R, t


def compute_ate(gt_xyz, es_aligned):
    """Compute ATE RMSE."""
    diff = gt_xyz - es_aligned
    dists = np.linalg.norm(diff, axis=1)
    return np.sqrt(np.mean(dists ** 2))


METHOD_DISPLAY = {
    "LoGeR": ("LoGeR", "#2196F3", "-"),
    "LoGeR_star_se3": ("LoGeR* (SE3)", "#FF5722", "--"),
    "LoGeR_star_sim3": ("LoGeR* (Sim3)", "#4CAF50", "-."),
}

SEQS = [f"{i:02d}" for i in range(11)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--result_base", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_ate = {m: {} for m in args.methods}

    for seq in SEQS:
        gt_file = os.path.join(args.gt_dir, f"{seq}.txt")
        if not os.path.exists(gt_file):
            print(f"[SKIP] GT not found: {gt_file}")
            continue
        gt_poses = read_kitti_poses(gt_file)
        gt_xyz = np.array([T[:3, 3] for T in gt_poses])

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        views = [
            ("Top (X-Z)", 0, 2),
            ("Front (X-Y)", 0, 1),
            ("Side (Z-Y)", 2, 1),
        ]

        has_any = False
        for method in args.methods:
            es_file = os.path.join(args.result_base, method, f"{seq}.txt")
            if not os.path.exists(es_file):
                print(f"  [SKIP] {method} seq {seq}: not found")
                continue

            es_poses = read_tum_poses(es_file)
            if len(es_poses) == 0:
                continue

            n = min(len(gt_xyz), len(es_poses))
            es_xyz = np.array([T[:3, 3] for T in es_poses[:n]])
            gt_sub = gt_xyz[:n]

            aligned, _, _, _ = align_sim3(gt_sub, es_xyz)
            ate = compute_ate(gt_sub, aligned)
            all_ate[method][seq] = ate

            display_name, color, ls = METHOD_DISPLAY.get(
                method, (method, "#999999", "-")
            )
            label = f"{display_name} (ATE={ate:.2f}m)"

            for ax, (_, d1, d2) in zip(axes, views):
                ax.plot(aligned[:, d1], aligned[:, d2],
                        color=color, linestyle=ls, linewidth=1.2,
                        label=label, alpha=0.85)
            has_any = True

        if not has_any:
            plt.close(fig)
            continue

        for ax, (title, d1, d2) in zip(axes, views):
            ax.plot(gt_xyz[:, d1], gt_xyz[:, d2],
                    color="black", linewidth=1.5, label="GT", alpha=0.7)
            ax.set_title(f"Seq {seq} - {title}", fontsize=13)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"KITTI Seq {seq}: Trajectory Comparison", fontsize=15, y=1.02)
        fig.tight_layout()
        out_path = os.path.join(args.output_dir, f"seq_{seq}_comparison.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # --- Summary bar chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SEQS))
    width = 0.25
    offsets = np.linspace(-width, width, len(args.methods))

    for i, method in enumerate(args.methods):
        ates = [all_ate[method].get(s, np.nan) for s in SEQS]
        display_name, color, _ = METHOD_DISPLAY.get(
            method, (method, "#999999", "-")
        )
        valid = [a for a in ates if not np.isnan(a)]
        mean_val = np.nanmean(valid) if valid else float("nan")
        bars = ax.bar(x + offsets[i], ates, width * 0.9,
                      label=f"{display_name} (avg={mean_val:.2f}m)",
                      color=color, alpha=0.8)

    ax.set_xlabel("Sequence", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_title("KITTI-od ATE Comparison: LoGeR vs LoGeR*(SE3) vs LoGeR*(Sim3)",
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(SEQS)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    summary_path = os.path.join(args.output_dir, "ate_summary_bar.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary: {summary_path}")

    # --- Sim3 scale drift plots ---
    sim3_method = "LoGeR_star_sim3"
    if sim3_method in args.methods:
        scale_seqs = {}
        for seq in SEQS:
            sf = os.path.join(args.result_base, sim3_method, f"{seq}.scale.txt")
            if os.path.exists(sf):
                rel, cum = [], []
                with open(sf) as f:
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            rel.append(float(parts[1]))
                            cum.append(float(parts[2]))
                if rel:
                    scale_seqs[seq] = {"relative": np.array(rel), "cumulative": np.array(cum)}

        if scale_seqs:
            n_seqs = len(scale_seqs)

            # --- Per-sequence scale figure ---
            ncols = min(4, n_seqs)
            nrows = (n_seqs + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

            for idx, (seq, data) in enumerate(sorted(scale_seqs.items())):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                windows = np.arange(len(data["cumulative"]))
                ax.plot(windows, data["cumulative"], color="#4CAF50", linewidth=1.5, label="Cumulative scale")
                ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.fill_between(windows, 1.0, data["cumulative"], alpha=0.15, color="#4CAF50")
                ax.set_title(f"Seq {seq} ({len(windows)} windows)", fontsize=11)
                ax.set_xlabel("Window index")
                ax.set_ylabel("Cumulative scale")
                ax.grid(True, alpha=0.3)
                final = data["cumulative"][-1]
                ax.annotate(f"final={final:.4f}", xy=(windows[-1], final),
                            fontsize=8, ha="right", va="bottom" if final > 1 else "top",
                            color="#D32F2F", fontweight="bold")

            for idx in range(n_seqs, nrows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].set_visible(False)

            fig.suptitle("LoGeR* (Sim3): Cumulative Scale Drift per Sequence", fontsize=14, y=1.01)
            fig.tight_layout()
            scale_per_seq = os.path.join(args.output_dir, "sim3_scale_drift_per_seq.png")
            fig.savefig(scale_per_seq, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {scale_per_seq}")

            # --- Summary: final cumulative scale bar chart ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            sorted_seqs = sorted(scale_seqs.keys())
            finals = [scale_seqs[s]["cumulative"][-1] for s in sorted_seqs]
            colors = ["#D32F2F" if abs(f - 1.0) > 0.1 else "#4CAF50" for f in finals]
            bars = ax1.bar(sorted_seqs, finals, color=colors, alpha=0.8, edgecolor="gray")
            ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax1.set_xlabel("Sequence", fontsize=12)
            ax1.set_ylabel("Final cumulative scale", fontsize=12)
            ax1.set_title("Final Cumulative Scale (ideal = 1.0)", fontsize=13)
            ax1.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, finals):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

            for seq in sorted_seqs:
                data = scale_seqs[seq]
                ax2.plot(np.arange(len(data["cumulative"])), data["cumulative"],
                         linewidth=1.2, label=f"Seq {seq}", alpha=0.8)
            ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax2.set_xlabel("Window index", fontsize=12)
            ax2.set_ylabel("Cumulative scale", fontsize=12)
            ax2.set_title("Scale Drift Across All Sequences", fontsize=13)
            ax2.legend(fontsize=7, loc="best", ncol=2)
            ax2.grid(True, alpha=0.3)

            fig.suptitle("LoGeR* (Sim3): Scale Drift Analysis", fontsize=15, y=1.02)
            fig.tight_layout()
            scale_summary = os.path.join(args.output_dir, "sim3_scale_drift_summary.png")
            fig.savefig(scale_summary, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {scale_summary}")

    # --- Collect timing from .timing.txt files ---
    all_timing = {m: {} for m in args.methods}
    for method in args.methods:
        for seq in SEQS:
            timing_file = os.path.join(args.result_base, method, f"{seq}.timing.txt")
            if os.path.exists(timing_file):
                with open(timing_file) as f:
                    for line in f:
                        if line.startswith("inference_time_s:"):
                            all_timing[method][seq] = float(line.split(":")[1].strip())

    # --- Print ATE table ---
    print("\n" + "=" * 90)
    print(f"{'Seq':<6}{'Frames':<8}", end="")
    for m in args.methods:
        dn, _, _ = METHOD_DISPLAY.get(m, (m, "", ""))
        print(f"{dn:<20}", end="")
    print()
    print("-" * 90)

    seq_frames = {
        "00": 4541, "01": 1101, "02": 4661, "03": 801, "04": 271,
        "05": 2761, "06": 1101, "07": 1101, "08": 4071, "09": 1591,
        "10": 1201,
    }

    for s in SEQS:
        nf = seq_frames.get(s, "?")
        print(f"{s:<6}{nf:<8}", end="")
        for m in args.methods:
            v = all_ate[m].get(s, float("nan"))
            print(f"{v:<20.2f}" if not np.isnan(v) else f"{'N/A':<20}", end="")
        print()
    print("-" * 90)
    print(f"{'AVG':<14}", end="")
    for m in args.methods:
        vals = [all_ate[m][s] for s in SEQS if s in all_ate[m]]
        avg = np.mean(vals) if vals else float("nan")
        print(f"{avg:<20.2f}" if not np.isnan(avg) else f"{'N/A':<20}", end="")
    print()
    print("=" * 90)

    # --- Print timing table ---
    has_timing = any(all_timing[m] for m in args.methods)
    if has_timing:
        print("\n" + "=" * 90)
        print("Inference Timing (torch.cuda.Event, excludes data loading)")
        print("-" * 90)
        print(f"{'Seq':<6}{'Frames':<8}", end="")
        for m in args.methods:
            dn, _, _ = METHOD_DISPLAY.get(m, (m, "", ""))
            print(f"{dn + ' (s)':<20}", end="")
        print()
        print("-" * 90)
        for s in SEQS:
            nf = seq_frames.get(s, "?")
            print(f"{s:<6}{nf:<8}", end="")
            for m in args.methods:
                t = all_timing[m].get(s, float("nan"))
                print(f"{t:<20.1f}" if not np.isnan(t) else f"{'N/A':<20}", end="")
            print()
        print("-" * 90)
        print(f"{'TOTAL':<14}", end="")
        for m in args.methods:
            vals = [all_timing[m][s] for s in SEQS if s in all_timing[m]]
            total = sum(vals) if vals else float("nan")
            print(f"{total:<20.1f}" if not np.isnan(total) else f"{'N/A':<20}", end="")
        print()
        total_frames = sum(seq_frames[s] for s in SEQS)
        print(f"{'FPS':<14}", end="")
        for m in args.methods:
            vals = [all_timing[m][s] for s in SEQS if s in all_timing[m]]
            total = sum(vals) if vals else 0
            frames = sum(seq_frames[s] for s in SEQS if s in all_timing[m])
            fps = frames / total if total > 0 else float("nan")
            print(f"{fps:<20.2f}" if not np.isnan(fps) else f"{'N/A':<20}", end="")
        print()
        print("=" * 90)


if __name__ == "__main__":
    main()
