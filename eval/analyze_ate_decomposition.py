"""
ATE Error Decomposition: Scale vs Pose for LoGeR* SE3 on KITTI.

Decomposes ATE into three components:
  1. Intra-chunk pose error  (per-chunk Sim3 aligned, the irreducible floor)
  2. Inter-chunk scale error  (fixed by oracle per-chunk scale)
  3. Inter-chunk pose error   (rotation / translation-direction across chunks)

Usage:
  python eval/analyze_ate_decomposition.py \
      --gt   data/kitti/dataset/poses/00.txt \
      --se3  results/viser_pi3_kitti/LoGeR_star_se3/00.txt \
      --sim3 results/viser_pi3_kitti/LoGeR_star_sim3/00.txt \
      --window_size 32 --overlap_size 3 \
      --output_dir results/viser_pi3_kitti/ate_decomposition_00
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers (same conventions as plot_kitti_comparison.py)
# ---------------------------------------------------------------------------

def read_kitti_poses(filepath):
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
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------

def align_sim3(gt_xyz, es_xyz):
    """Sim(3) Umeyama alignment. Returns (aligned, scale, R, t)."""
    n = gt_xyz.shape[0]
    mu_gt = gt_xyz.mean(axis=0)
    mu_es = es_xyz.mean(axis=0)
    gt_c = gt_xyz - mu_gt
    es_c = es_xyz - mu_es

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


def align_se3(gt_xyz, es_xyz):
    """SE(3) Umeyama alignment (scale=1). Returns (aligned, R, t)."""
    n = gt_xyz.shape[0]
    mu_gt = gt_xyz.mean(axis=0)
    mu_es = es_xyz.mean(axis=0)
    gt_c = gt_xyz - mu_gt
    es_c = es_xyz - mu_es

    cov = (gt_c.T @ es_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    t = mu_gt - R @ mu_es
    aligned = (es_xyz @ R.T) + t
    return aligned, R, t


def compute_ate(gt_xyz, es_aligned):
    diff = gt_xyz - es_aligned
    dists = np.linalg.norm(diff, axis=1)
    return np.sqrt(np.mean(dists ** 2)), dists


def rotation_error_deg(R):
    """Geodesic rotation error in degrees."""
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


# ---------------------------------------------------------------------------
# Chunk utilities
# ---------------------------------------------------------------------------

def compute_chunk_boundaries(N, window_size, overlap_size):
    """Reproduce the windowing logic from pi3.py."""
    step = max(window_size - overlap_size, 1)
    windows = []
    for start_idx in range(0, N, step):
        end_idx = min(start_idx + window_size, N)
        if end_idx - start_idx >= overlap_size or (end_idx == N and start_idx < N):
            windows.append((start_idx, end_idx))
        if end_idx == N:
            break
    return windows


def chunk_owned_frames(windows):
    """
    For each chunk, determine the frames it 'owns' after overlap removal.
    Chunk 0 owns [0, stride), chunk 1 owns [stride, 2*stride), ...
    Last chunk owns up to the end.
    """
    owned = []
    for i, (s, e) in enumerate(windows):
        if i == 0:
            start_own = s
        else:
            start_own = windows[i-1][1] - (windows[i-1][1] - s)
            start_own = s  # chunk i starts at s in raw, but after merge...
        owned.append((s, e))
    # After SE3 merge the non-overlap owned regions are:
    # chunk 0: frames [0, stride)
    # chunk i: frames [i*stride, (i+1)*stride)   (last chunk gets remainder)
    n_chunks = len(windows)
    if n_chunks <= 1:
        return [(windows[0][0], windows[0][1])]
    stride = windows[1][0] - windows[0][0]
    result = []
    for i in range(n_chunks):
        f_start = i * stride
        if i < n_chunks - 1:
            f_end = (i + 1) * stride
        else:
            f_end = windows[-1][1]
        result.append((f_start, f_end))
    return result


# ---------------------------------------------------------------------------
# Oracle scale: build a piecewise-rescaled trajectory
# ---------------------------------------------------------------------------

def build_oracle_scale_trajectory(est_xyz, gt_xyz, owned_ranges):
    """
    For each chunk, compute oracle scale = GT_path_length / EST_path_length.
    Then rebuild the trajectory by rescaling each chunk's incremental
    displacements by the oracle scale ratio.
    Returns: (corrected_xyz, per_chunk_scales)
    """
    n_chunks = len(owned_ranges)
    chunk_scales_gt = np.ones(n_chunks)
    chunk_scales_est = np.ones(n_chunks)

    for ci, (fs, fe) in enumerate(owned_ranges):
        if fe - fs < 2:
            chunk_scales_gt[ci] = 1.0
            chunk_scales_est[ci] = 1.0
            continue
        gt_disp = np.linalg.norm(np.diff(gt_xyz[fs:fe], axis=0), axis=1)
        est_disp = np.linalg.norm(np.diff(est_xyz[fs:fe], axis=0), axis=1)
        chunk_scales_gt[ci] = gt_disp.sum()
        chunk_scales_est[ci] = est_disp.sum()

    oracle_scales = np.where(
        chunk_scales_est > 1e-8,
        chunk_scales_gt / chunk_scales_est,
        1.0,
    )

    corrected = np.zeros_like(est_xyz)
    corrected[0] = est_xyz[0]
    for ci, (fs, fe) in enumerate(owned_ranges):
        s = oracle_scales[ci]
        for j in range(max(fs, 1) if ci == 0 else fs, fe):
            prev = j - 1 if j > 0 else 0
            corrected[j] = corrected[prev] + s * (est_xyz[j] - est_xyz[prev])

    return corrected, oracle_scales


def build_piecewise_aligned_trajectory(est_xyz, gt_xyz, est_poses, gt_poses, owned_ranges):
    """
    Per-chunk independent Sim(3) alignment: each chunk aligned to GT separately.
    Returns per-frame aligned positions + per-frame errors.
    """
    N = est_xyz.shape[0]
    aligned = np.zeros_like(est_xyz)
    per_chunk_ate = []
    per_chunk_rot_err = []

    for ci, (fs, fe) in enumerate(owned_ranges):
        if fe - fs < 3:
            aligned[fs:fe] = gt_xyz[fs:fe]
            per_chunk_ate.append(0.0)
            per_chunk_rot_err.append(0.0)
            continue
        chunk_gt = gt_xyz[fs:fe]
        chunk_est = est_xyz[fs:fe]
        chunk_aligned, s, R, t = align_sim3(chunk_gt, chunk_est)
        aligned[fs:fe] = chunk_aligned

        _, dists = compute_ate(chunk_gt, chunk_aligned)
        per_chunk_ate.append(np.sqrt(np.mean(dists**2)))

        rot_errs = []
        for j in range(fe - fs):
            R_gt = gt_poses[fs + j][:3, :3]
            R_est = est_poses[fs + j][:3, :3]
            R_aligned = R @ R_est
            R_err = R_gt.T @ R_aligned
            rot_errs.append(rotation_error_deg(R_err))
        per_chunk_rot_err.append(np.mean(rot_errs))

    return aligned, np.array(per_chunk_ate), np.array(per_chunk_rot_err)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ATE Error Decomposition")
    parser.add_argument("--gt", required=True, help="KITTI GT poses (3x4)")
    parser.add_argument("--se3", required=True, help="LoGeR* SE3 TUM trajectory")
    parser.add_argument("--sim3", default=None, help="LoGeR* Sim3 TUM trajectory (optional)")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--overlap_size", type=int, default=3)
    parser.add_argument("--output_dir", default="results/ate_decomposition")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    gt_poses = read_kitti_poses(args.gt)
    se3_poses = read_tum_poses(args.se3)
    N = min(len(gt_poses), len(se3_poses))
    gt_poses = gt_poses[:N]
    se3_poses = se3_poses[:N]
    gt_xyz = np.array([T[:3, 3] for T in gt_poses])
    se3_xyz = np.array([T[:3, 3] for T in se3_poses])

    sim3_poses_list = None
    sim3_xyz = None
    if args.sim3 and os.path.exists(args.sim3):
        sim3_poses_list = read_tum_poses(args.sim3)[:N]
        sim3_xyz = np.array([T[:3, 3] for T in sim3_poses_list])

    print(f"Loaded {N} frames")

    # --- Chunk boundaries ---
    windows = compute_chunk_boundaries(N, args.window_size, args.overlap_size)
    owned = chunk_owned_frames(windows)
    n_chunks = len(owned)
    stride = max(windows[1][0] - windows[0][0], 1) if len(windows) > 1 else N
    print(f"Chunks: {n_chunks}, window={args.window_size}, overlap={args.overlap_size}, stride={stride}")

    # =====================================================================
    # Metric 1: ATE_SE3 — global Sim(3) alignment (standard baseline)
    # =====================================================================
    se3_aligned, se3_global_scale, _, _ = align_sim3(gt_xyz, se3_xyz)
    ate_se3, se3_per_frame_err = compute_ate(gt_xyz, se3_aligned)
    print(f"\n[1] ATE_SE3 (global Sim3 align):  {ate_se3:.4f} m  (global_scale={se3_global_scale:.4f})")

    # =====================================================================
    # Metric 2: ATE_oracle — oracle per-chunk scale + global Sim(3)
    # =====================================================================
    # First globally align SE3 to get a reasonable coordinate frame,
    # then compute per-chunk oracle scales on the aligned trajectory.
    oracle_corrected, oracle_scales = build_oracle_scale_trajectory(
        se3_aligned, gt_xyz, owned
    )
    # Re-align globally after oracle correction
    oracle_realigned, oracle_global_s, _, _ = align_sim3(gt_xyz, oracle_corrected)
    ate_oracle, oracle_per_frame_err = compute_ate(gt_xyz, oracle_realigned)
    print(f"[2] ATE_oracle (per-chunk GT scale): {ate_oracle:.4f} m")

    # =====================================================================
    # Metric 3: ATE_per_chunk — each chunk independently Sim(3) aligned
    # =====================================================================
    piecewise_aligned, per_chunk_ate, per_chunk_rot = build_piecewise_aligned_trajectory(
        se3_xyz, gt_xyz, se3_poses, gt_poses, owned
    )
    ate_piecewise, piecewise_per_frame_err = compute_ate(gt_xyz, piecewise_aligned)
    print(f"[3] ATE_per_chunk (independent Sim3): {ate_piecewise:.4f} m")

    # =====================================================================
    # Metric 4 (optional): ATE_Sim3 — LoGeR* Sim3 merge result
    # =====================================================================
    ate_sim3 = None
    if sim3_xyz is not None:
        sim3_aligned, sim3_gs, _, _ = align_sim3(gt_xyz, sim3_xyz)
        ate_sim3_val, sim3_per_frame_err = compute_ate(gt_xyz, sim3_aligned)
        ate_sim3 = ate_sim3_val
        print(f"[4] ATE_Sim3 (depth-based scale):  {ate_sim3:.4f} m  (global_scale={sim3_gs:.4f})")

    # =====================================================================
    # Error decomposition
    # =====================================================================
    scale_contribution = ate_se3 - ate_oracle
    pose_stitching_contribution = ate_oracle - ate_piecewise
    intra_chunk_error = ate_piecewise

    print(f"\n{'='*65}")
    print(f"ATE Error Decomposition for KITTI Seq 00")
    print(f"{'='*65}")
    print(f"  Total ATE (SE3):           {ate_se3:8.4f} m  (100.0%)")
    print(f"  ├─ Scale contribution:     {scale_contribution:8.4f} m  ({100*scale_contribution/ate_se3:5.1f}%)")
    print(f"  ├─ Pose-stitching error:   {pose_stitching_contribution:8.4f} m  ({100*pose_stitching_contribution/ate_se3:5.1f}%)")
    print(f"  └─ Intra-chunk error:      {intra_chunk_error:8.4f} m  ({100*intra_chunk_error/ate_se3:5.1f}%)")
    if ate_sim3 is not None:
        print(f"  [ref] ATE_Sim3 (model):    {ate_sim3:8.4f} m")
    print(f"{'='*65}")

    # =====================================================================
    # Per-chunk oracle scale analysis
    # =====================================================================
    print(f"\nPer-chunk oracle scale statistics:")
    print(f"  Mean:   {oracle_scales.mean():.4f}")
    print(f"  Std:    {oracle_scales.std():.4f}")
    print(f"  Min:    {oracle_scales.min():.4f} (chunk {oracle_scales.argmin()})")
    print(f"  Max:    {oracle_scales.max():.4f} (chunk {oracle_scales.argmax()})")
    print(f"  CV:     {oracle_scales.std()/oracle_scales.mean():.4f} (coefficient of variation)")
    # Scale relative changes
    rel_scale_changes = oracle_scales[1:] / oracle_scales[:-1]
    print(f"  Relative scale changes: mean={rel_scale_changes.mean():.4f}, std={rel_scale_changes.std():.4f}")

    # =====================================================================
    # Visualization
    # =====================================================================

    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # --- (1) Trajectory top-view comparison ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k-', lw=1.5, label='GT', alpha=0.7)
    ax.plot(se3_aligned[:, 0], se3_aligned[:, 2], 'r-', lw=1, label=f'SE3 (ATE={ate_se3:.2f}m)', alpha=0.7)
    ax.plot(oracle_realigned[:, 0], oracle_realigned[:, 2], 'b-', lw=1, label=f'Oracle-scale (ATE={ate_oracle:.2f}m)', alpha=0.7)
    ax.plot(piecewise_aligned[:, 0], piecewise_aligned[:, 2], 'g-', lw=1, label=f'Per-chunk (ATE={ate_piecewise:.2f}m)', alpha=0.7)
    if sim3_xyz is not None:
        ax.plot(sim3_aligned[:, 0], sim3_aligned[:, 2], color='orange', ls='--', lw=1,
                label=f'Sim3 (ATE={ate_sim3:.2f}m)', alpha=0.7)
    ax.set_title("Top View (X-Z)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # --- (2) Per-frame ATE comparison ---
    ax = fig.add_subplot(gs[0, 1:])
    frames = np.arange(N)
    ax.plot(frames, se3_per_frame_err, 'r-', lw=0.6, label=f'SE3 (RMSE={ate_se3:.2f}m)', alpha=0.7)
    ax.plot(frames, oracle_per_frame_err, 'b-', lw=0.6, label=f'Oracle-scale (RMSE={ate_oracle:.2f}m)', alpha=0.7)
    ax.plot(frames, piecewise_per_frame_err, 'g-', lw=0.6, label=f'Per-chunk (RMSE={ate_piecewise:.2f}m)', alpha=0.7)
    # Mark chunk boundaries
    for ci, (fs, fe) in enumerate(owned):
        if ci > 0:
            ax.axvline(x=fs, color='gray', ls=':', lw=0.3, alpha=0.5)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Position Error (m)", fontsize=11)
    ax.set_title("Per-frame ATE: SE3 vs Oracle-scale vs Per-chunk", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (3) Per-chunk oracle scale ---
    ax = fig.add_subplot(gs[1, 0:2])
    chunk_indices = np.arange(n_chunks)
    colors = plt.cm.RdYlGn(np.clip(1.0 - np.abs(oracle_scales - oracle_scales.mean()) / oracle_scales.mean(), 0, 1))
    ax.bar(chunk_indices, oracle_scales, color=colors, edgecolor='gray', lw=0.3, alpha=0.8)
    ax.axhline(y=oracle_scales.mean(), color='blue', ls='--', lw=1, label=f'Mean={oracle_scales.mean():.3f}')
    ax.axhline(y=1.0, color='black', ls=':', lw=0.8, alpha=0.5, label='1.0 (ideal if consistent)')
    ax.set_xlabel("Chunk Index", fontsize=11)
    ax.set_ylabel("Oracle Scale (GT/EST path ratio)", fontsize=11)
    ax.set_title(f"Per-chunk Oracle Scale (std={oracle_scales.std():.4f}, CV={oracle_scales.std()/oracle_scales.mean():.3f})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # --- (4) Error decomposition bar chart ---
    ax = fig.add_subplot(gs[1, 2])
    labels = ['Total\n(SE3)', 'Scale\nContrib.', 'Pose-stitch\nContrib.', 'Intra-chunk\nError']
    values = [ate_se3, scale_contribution, pose_stitching_contribution, intra_chunk_error]
    bar_colors = ['#D32F2F', '#FF9800', '#2196F3', '#4CAF50']
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='gray', alpha=0.85)
    for bar, val in zip(bars, values):
        pct = 100 * val / ate_se3 if ate_se3 > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*ate_se3,
                f'{val:.2f}m\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("ATE (m)", fontsize=11)
    ax.set_title("ATE Error Decomposition", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # --- (5) Oracle scale mapped to trajectory ---
    ax = fig.add_subplot(gs[2, 0:2])
    frame_scale = np.ones(N)
    for ci, (fs, fe) in enumerate(owned):
        frame_scale[fs:fe] = oracle_scales[ci]
    sc = ax.scatter(gt_xyz[:, 0], gt_xyz[:, 2], c=frame_scale, cmap='coolwarm',
                    s=1, alpha=0.8, vmin=oracle_scales.min(), vmax=oracle_scales.max())
    plt.colorbar(sc, ax=ax, label='Oracle Scale', shrink=0.8)
    ax.set_title("GT Trajectory colored by per-chunk Oracle Scale", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- (6) Per-chunk ATE (intra-chunk) ---
    ax = fig.add_subplot(gs[2, 2])
    ax.bar(chunk_indices, per_chunk_ate, color='#4CAF50', edgecolor='gray', lw=0.3, alpha=0.8)
    ax.set_xlabel("Chunk Index", fontsize=11)
    ax.set_ylabel("ATE (m)", fontsize=11)
    ax.set_title(f"Per-chunk ATE (mean={per_chunk_ate.mean():.3f}m)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # --- (7) Scale variation vs per-chunk ATE scatter ---
    ax = fig.add_subplot(gs[3, 0])
    scale_dev = np.abs(oracle_scales - oracle_scales.mean())
    ax.scatter(scale_dev, per_chunk_ate, alpha=0.6, s=15, c=chunk_indices, cmap='viridis')
    ax.set_xlabel("|Oracle Scale - Mean Scale|", fontsize=11)
    ax.set_ylabel("Per-chunk ATE (m)", fontsize=11)
    ax.set_title("Scale Deviation vs Intra-chunk Error", fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- (8) Cumulative error growth ---
    ax = fig.add_subplot(gs[3, 1:])
    # Sliding window average of per-frame error
    win = max(stride, 10)
    se3_smooth = np.convolve(se3_per_frame_err, np.ones(win)/win, mode='valid')
    oracle_smooth = np.convolve(oracle_per_frame_err, np.ones(win)/win, mode='valid')
    pw_smooth = np.convolve(piecewise_per_frame_err, np.ones(win)/win, mode='valid')
    ax.fill_between(np.arange(len(se3_smooth)), pw_smooth, se3_smooth,
                    alpha=0.15, color='red', label='Gap (scale + pose-stitch)')
    ax.fill_between(np.arange(len(oracle_smooth)), pw_smooth, oracle_smooth,
                    alpha=0.2, color='blue', label='Gap (pose-stitch only)')
    ax.plot(se3_smooth, 'r-', lw=1, label='SE3', alpha=0.8)
    ax.plot(oracle_smooth, 'b-', lw=1, label='Oracle-scale', alpha=0.8)
    ax.plot(pw_smooth, 'g-', lw=1, label='Per-chunk', alpha=0.8)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel(f"Smoothed ATE (window={win})", fontsize=11)
    ax.set_title("Error Growth: Scale gap (red fill) vs Pose-stitch gap (blue fill)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KITTI Seq 00: ATE Error Decomposition — Scale vs Pose", fontsize=16, y=0.98)
    out_path = os.path.join(args.output_dir, "ate_decomposition.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # =====================================================================
    # Save numerical summary
    # =====================================================================
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ATE Error Decomposition — KITTI Seq 00\n")
        f.write(f"{'='*60}\n")
        f.write(f"Frames: {N}, Chunks: {n_chunks}, Window: {args.window_size}, Overlap: {args.overlap_size}\n\n")
        f.write(f"ATE_SE3 (global Sim3):         {ate_se3:.4f} m  (100.0%)\n")
        f.write(f"ATE_oracle (per-chunk scale):   {ate_oracle:.4f} m  ({100*ate_oracle/ate_se3:.1f}%)\n")
        f.write(f"ATE_per_chunk (independent):    {ate_piecewise:.4f} m  ({100*ate_piecewise/ate_se3:.1f}%)\n")
        if ate_sim3 is not None:
            f.write(f"ATE_Sim3 (model scale):        {ate_sim3:.4f} m\n")
        f.write(f"\nDecomposition:\n")
        f.write(f"  Scale contribution:     {scale_contribution:.4f} m  ({100*scale_contribution/ate_se3:.1f}%)\n")
        f.write(f"  Pose-stitch contribution: {pose_stitching_contribution:.4f} m  ({100*pose_stitching_contribution/ate_se3:.1f}%)\n")
        f.write(f"  Intra-chunk error:      {intra_chunk_error:.4f} m  ({100*intra_chunk_error/ate_se3:.1f}%)\n")
        f.write(f"\nOracle scale stats:\n")
        f.write(f"  Mean={oracle_scales.mean():.4f}, Std={oracle_scales.std():.4f}, CV={oracle_scales.std()/oracle_scales.mean():.4f}\n")
        f.write(f"  Min={oracle_scales.min():.4f} (chunk {oracle_scales.argmin()}), Max={oracle_scales.max():.4f} (chunk {oracle_scales.argmax()})\n")

        f.write(f"\nPer-chunk details:\n")
        f.write(f"{'Chunk':>6} {'Frames':>12} {'OracleScale':>12} {'ChunkATE':>10} {'RotErr(deg)':>12}\n")
        for ci, (fs, fe) in enumerate(owned):
            f.write(f"{ci:>6} {f'{fs}-{fe-1}':>12} {oracle_scales[ci]:>12.4f} {per_chunk_ate[ci]:>10.4f} {per_chunk_rot[ci]:>12.4f}\n")

    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
