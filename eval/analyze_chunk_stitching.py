"""
Per-boundary chunk stitching error analysis for LoGeR* SE3 on KITTI.

Produces TWO result figures:
  Figure 1 — Local stitching error: how badly each boundary is stitched
              (per-chunk independent alignment, ignoring accumulated drift).
  Figure 2 — Cumulative error impact: how much each boundary contributes
              to overall trajectory drift (considering position in sequence).

Usage:
  python eval/analyze_chunk_stitching.py \
      --gt   data/kitti/dataset/poses/00.txt \
      --se3  results/viser_pi3_kitti/LoGeR_star_se3/00.txt \
      --window_size 32 --overlap_size 3 \
      --output_dir results/viser_pi3_kitti/ate_decomposition_00 \
      --top_k 50
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ── I/O ──────────────────────────────────────────────────────────────────

def read_kitti_poses(fp):
    poses = []
    with open(fp) as f:
        for line in f:
            v = list(map(float, line.strip().split()))
            if len(v) == 12:
                T = np.eye(4); T[:3, :] = np.array(v).reshape(3, 4)
                poses.append(T)
    return poses

def quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])

def read_tum_poses(fp):
    poses = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            v = list(map(float, line.split()))
            if len(v) >= 8:
                _, tx, ty, tz, qx, qy, qz, qw = v[:8]
                T = np.eye(4); T[:3, 3] = [tx, ty, tz]
                T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
                poses.append(T)
    return poses


# ── geometry helpers ─────────────────────────────────────────────────────

def align_sim3(gt, es):
    n = gt.shape[0]; mu_g = gt.mean(0); mu_e = es.mean(0)
    gc = gt - mu_g; ec = es - mu_e
    sig_e = np.sum(ec**2) / n
    cov = gc.T @ ec / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: S[2,2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sig_e
    t = mu_g - s * R @ mu_e
    return s * (es @ R.T) + t, s, R, t

def rot_err_deg(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ── chunk logic ──────────────────────────────────────────────────────────

def compute_windows(N, ws, ov):
    step = max(ws - ov, 1); wins = []
    for s in range(0, N, step):
        e = min(s + ws, N)
        if e - s >= ov or (e == N and s < N): wins.append((s, e))
        if e == N: break
    return wins

def owned_ranges(wins):
    nc = len(wins)
    if nc <= 1: return [(wins[0][0], wins[0][1])]
    stride = wins[1][0] - wins[0][0]; res = []
    for i in range(nc):
        fs = i * stride
        fe = (i+1)*stride if i < nc-1 else wins[-1][1]
        res.append((fs, fe))
    return res


# ── per-boundary analysis ────────────────────────────────────────────────

def analyse_boundaries(gt_poses, se3_poses, windows, owned, gt_xyz, se3_xyz):
    """
    For each boundary between consecutive chunks, compute:
      local_rot_err   — rotation discrepancy between the two per-chunk-aligned
                         poses at the overlap frame (degrees)
      local_trans_err — translation discrepancy (metres) at the overlap frame
      cumul_err_jump  — increase in per-frame position error (in globally-
                         aligned SE3 trajectory) across the boundary
    """
    n_chunks = len(owned)
    N = gt_xyz.shape[0]

    # global Sim3 aligned SE3
    se3_al, _, R_g, t_g = align_sim3(gt_xyz, se3_xyz)
    per_frame_err = np.linalg.norm(gt_xyz - se3_al, axis=1)

    # per-chunk independent Sim3 alignment + poses
    chunk_aligned_xyz = [None] * n_chunks
    chunk_aligned_R   = [None] * n_chunks  # per-frame aligned rotation
    for ci, (ws, we) in enumerate(windows):
        cg = gt_xyz[ws:we]; ce = se3_xyz[ws:we]
        if len(cg) < 3:
            chunk_aligned_xyz[ci] = cg.copy()
            chunk_aligned_R[ci] = np.array([p[:3,:3] for p in gt_poses[ws:we]])
            continue
        al, s, R, t = align_sim3(cg, ce)
        chunk_aligned_xyz[ci] = al
        chunk_aligned_R[ci] = np.array([R @ se3_poses[j][:3,:3] for j in range(ws, we)])

    stride = windows[1][0] - windows[0][0] if len(windows) > 1 else N
    overlap = windows[0][1] - windows[1][0] if len(windows) > 1 else 0

    records = []
    for bi in range(n_chunks - 1):
        ws_a, we_a = windows[bi]
        ws_b, we_b = windows[bi + 1]
        # overlap frames are [ws_b, we_a)
        ov_start = ws_b
        ov_end   = we_a
        if ov_start >= ov_end:
            ov_start = ov_end - 1  # fallback
        mid_ov = (ov_start + ov_end) // 2  # representative overlap frame

        # --- local error (per-chunk independent alignment) ---
        idx_in_a = mid_ov - ws_a
        idx_in_b = mid_ov - ws_b
        if idx_in_a < 0 or idx_in_a >= len(chunk_aligned_xyz[bi]):
            idx_in_a = len(chunk_aligned_xyz[bi]) - 1
        if idx_in_b < 0 or idx_in_b >= len(chunk_aligned_xyz[bi+1]):
            idx_in_b = 0

        pos_a = chunk_aligned_xyz[bi][idx_in_a]
        pos_b = chunk_aligned_xyz[bi+1][idx_in_b]
        local_trans_err = np.linalg.norm(pos_a - pos_b)

        R_a = chunk_aligned_R[bi][idx_in_a]
        R_b = chunk_aligned_R[bi+1][idx_in_b]
        R_gt_frame = gt_poses[mid_ov][:3, :3]
        # rotation discrepancy: how different are the two chunks' aligned
        # rotations relative to GT at the overlap frame
        local_rot_err_a = rot_err_deg(R_gt_frame.T @ R_a)
        local_rot_err_b = rot_err_deg(R_gt_frame.T @ R_b)
        local_rot_err = rot_err_deg(R_a.T @ R_b)  # direct disagreement

        # --- cumulative impact ---
        # average error in the owned region of chunk i+1  minus  chunk i
        fs_a, fe_a = owned[bi]
        fs_b, fe_b = owned[bi + 1]
        avg_err_a = per_frame_err[fs_a:fe_a].mean()
        avg_err_b = per_frame_err[fs_b:fe_b].mean()
        cumul_err_jump = avg_err_b - avg_err_a

        # max error in chunk i+1's region (captures the worst-case impact)
        max_err_b = per_frame_err[fs_b:fe_b].max()

        records.append(dict(
            boundary_idx=bi,
            chunk_a=bi, chunk_b=bi+1,
            frame=mid_ov,
            frame_range_a=(fs_a, fe_a),
            frame_range_b=(fs_b, fe_b),
            local_rot_err=local_rot_err,
            local_trans_err=local_trans_err,
            local_rot_err_a=local_rot_err_a,
            local_rot_err_b=local_rot_err_b,
            cumul_err_jump=cumul_err_jump,
            avg_err_before=avg_err_a,
            avg_err_after=avg_err_b,
            max_err_after=max_err_b,
        ))

    return records, se3_al, per_frame_err


# ── plotting ─────────────────────────────────────────────────────────────

def plot_local_errors(records, gt_xyz, se3_al, top_k, output_dir):
    """Figure 1: local stitching error (rotation + translation) per boundary."""
    # combine rotation and translation into a single score
    for r in records:
        r["local_combined"] = r["local_rot_err"] + r["local_trans_err"]

    sorted_rot = sorted(records, key=lambda r: r["local_rot_err"], reverse=True)
    top_rot = sorted_rot[:top_k]

    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ---- (a) bar chart: rotation error ----
    ax = fig.add_subplot(gs[0, :])
    idxs = [r["boundary_idx"] for r in top_rot]
    vals = [r["local_rot_err"] for r in top_rot]
    colors = plt.cm.Reds(np.linspace(0.9, 0.3, len(top_rot)))
    bars = ax.barh(range(len(top_rot)), vals, color=colors, edgecolor='gray', lw=0.4)
    ax.set_yticks(range(len(top_rot)))
    ax.set_yticklabels([f"Chunk {r['boundary_idx']}→{r['boundary_idx']+1}  (f{r['frame']})" for r in top_rot],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Local Rotation Error at Overlap (deg)", fontsize=11)
    ax.set_title(f"Top {top_k} Worst LOCAL Stitching Errors (rotation discrepancy between adjacent chunks)", fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}°", va='center', fontsize=7, color='#333')

    # ---- (b) trajectory with top-K boundaries highlighted ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k-', lw=1, alpha=0.4, label='GT')
    ax.plot(se3_al[:, 0], se3_al[:, 2], 'r-', lw=0.6, alpha=0.4, label='SE3 aligned')
    cmap = plt.cm.Reds
    for rank, r in enumerate(top_rot[:30]):
        f = r["frame"]
        intensity = 1.0 - rank / 30
        ax.plot(gt_xyz[f, 0], gt_xyz[f, 2], 'o', color=cmap(0.3 + 0.6*intensity),
                ms=5 + 4*intensity, zorder=5, alpha=0.8)
        ax.annotate(f"{r['boundary_idx']}", (gt_xyz[f, 0], gt_xyz[f, 2]),
                    fontsize=6, fontweight='bold', color='darkred',
                    xytext=(4, 4), textcoords='offset points')
    ax.set_title("Top 30 worst boundaries on GT trajectory (numbered)", fontsize=11)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (c) all boundaries: rotation error vs chunk index ----
    ax = fig.add_subplot(gs[1, 1])
    all_bi = [r["boundary_idx"] for r in records]
    all_rot = [r["local_rot_err"] for r in records]
    ax.bar(all_bi, all_rot, width=1.0, color='salmon', edgecolor='none', alpha=0.7)
    # highlight top-K
    for r in top_rot:
        ax.bar(r["boundary_idx"], r["local_rot_err"], width=1.0, color='red', edgecolor='none')
    ax.set_xlabel("Boundary Index (chunk i → i+1)", fontsize=11)
    ax.set_ylabel("Local Rotation Error (deg)", fontsize=11)
    ax.set_title("All boundaries: local rotation error (red = top-K)", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # ---- (d) translation error for top-K ----
    ax = fig.add_subplot(gs[2, 0])
    sorted_trans = sorted(records, key=lambda r: r["local_trans_err"], reverse=True)[:top_k]
    bars = ax.barh(range(len(sorted_trans)),
                   [r["local_trans_err"] for r in sorted_trans],
                   color=plt.cm.Oranges(np.linspace(0.9, 0.3, len(sorted_trans))),
                   edgecolor='gray', lw=0.4)
    ax.set_yticks(range(len(sorted_trans)))
    ax.set_yticklabels([f"Chunk {r['boundary_idx']}→{r['boundary_idx']+1}" for r in sorted_trans], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Local Translation Error at Overlap (m)", fontsize=11)
    ax.set_title(f"Top {top_k} Worst LOCAL Translation Discrepancy", fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # ---- (e) rotation error distribution histogram ----
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(all_rot, bins=50, color='salmon', edgecolor='gray', alpha=0.8)
    median_rot = np.median(all_rot)
    p90 = np.percentile(all_rot, 90)
    ax.axvline(median_rot, color='blue', ls='--', lw=1.5, label=f'Median={median_rot:.1f}°')
    ax.axvline(p90, color='red', ls='--', lw=1.5, label=f'P90={p90:.1f}°')
    ax.set_xlabel("Local Rotation Error (deg)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of local rotation errors across all boundaries", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Figure 1: LOCAL Stitching Error per Boundary (independent of accumulated drift)",
                 fontsize=15, y=0.98)
    out = os.path.join(output_dir, "chunk_stitching_local.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")
    return top_rot


def plot_cumulative_impact(records, gt_xyz, se3_al, per_frame_err, owned, top_k, output_dir):
    """Figure 2: cumulative error impact per boundary."""
    # use abs(cumul_err_jump) so large positive jumps rank highest
    sorted_cumul = sorted(records, key=lambda r: abs(r["cumul_err_jump"]), reverse=True)
    top_cumul = sorted_cumul[:top_k]

    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ---- (a) bar chart: cumulative error jump ----
    ax = fig.add_subplot(gs[0, :])
    vals = [r["cumul_err_jump"] for r in top_cumul]
    colors_arr = ['#D32F2F' if v > 0 else '#2196F3' for v in vals]
    bars = ax.barh(range(len(top_cumul)), vals, color=colors_arr, edgecolor='gray', lw=0.4, alpha=0.85)
    ax.set_yticks(range(len(top_cumul)))
    ax.set_yticklabels(
        [f"Chunk {r['boundary_idx']}→{r['boundary_idx']+1}  (f{r['frame']})  "
         f"[{r['avg_err_before']:.1f}→{r['avg_err_after']:.1f}m]" for r in top_cumul],
        fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Cumulative Error Jump: avg_ATE(chunk_{i+1}) − avg_ATE(chunk_i)  (m)", fontsize=10)
    ax.set_title(f"Top {top_k} Boundaries by CUMULATIVE Error Impact (red=error increase, blue=decrease)",
                 fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals):
        side = bar.get_width()
        ax.text(side + (0.3 if v >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                f"{v:+.2f}m", va='center', ha='left' if v >= 0 else 'right',
                fontsize=7, color='#333')

    # ---- (b) trajectory colored by per-frame error ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k-', lw=1.5, alpha=0.5, label='GT', zorder=1)
    sc = ax.scatter(se3_al[:, 0], se3_al[:, 2], c=per_frame_err, cmap='hot_r',
                    s=1, alpha=0.8, zorder=2)
    plt.colorbar(sc, ax=ax, label='Position Error (m)', shrink=0.8)
    for rank, r in enumerate(top_cumul[:30]):
        f = r["frame"]
        color = 'red' if r["cumul_err_jump"] > 0 else 'blue'
        ax.plot(se3_al[f, 0], se3_al[f, 2], 'o', color=color, ms=5, zorder=5, alpha=0.8)
        ax.annotate(f"{r['boundary_idx']}", (se3_al[f, 0], se3_al[f, 2]),
                    fontsize=6, fontweight='bold', color=color,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_title("Aligned SE3 trajectory colored by error, top-30 boundaries marked", fontsize=10)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # ---- (c) per-frame error curve with boundaries ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(per_frame_err, 'r-', lw=0.5, alpha=0.6, label='Per-frame error')
    # mark top-30 worst cumulative boundaries
    for rank, r in enumerate(top_cumul[:30]):
        f = r["frame"]
        color = 'red' if r["cumul_err_jump"] > 0 else 'blue'
        ax.axvline(f, color=color, ls='-', lw=0.8, alpha=0.5)
        ax.annotate(f"  {r['boundary_idx']}", (f, per_frame_err[min(f, len(per_frame_err)-1)]),
                    fontsize=6, color=color, fontweight='bold', rotation=90, va='bottom')
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Position Error (m)", fontsize=11)
    ax.set_title("Per-frame ATE with top-30 cumulative-impact boundaries", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (d) all boundaries: cumulative error jump ----
    ax = fig.add_subplot(gs[2, 0])
    all_bi = [r["boundary_idx"] for r in records]
    all_jump = [r["cumul_err_jump"] for r in records]
    bar_c = ['#D32F2F' if v > 0 else '#2196F3' for v in all_jump]
    ax.bar(all_bi, all_jump, width=1.0, color=bar_c, edgecolor='none', alpha=0.7)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("Boundary Index", fontsize=11)
    ax.set_ylabel("Error Jump (m)", fontsize=11)
    ax.set_title("All boundaries: cumulative error jump (red=increase, blue=decrease)", fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # ---- (e) average error per chunk (stacked area) ----
    ax = fig.add_subplot(gs[2, 1])
    chunk_avg_err = [per_frame_err[fs:fe].mean() for fs, fe in owned]
    chunk_max_err = [per_frame_err[fs:fe].max() for fs, fe in owned]
    ci_arr = np.arange(len(owned))
    ax.fill_between(ci_arr, 0, chunk_max_err, alpha=0.2, color='red', label='Max error in chunk')
    ax.fill_between(ci_arr, 0, chunk_avg_err, alpha=0.4, color='salmon', label='Avg error in chunk')
    ax.plot(ci_arr, chunk_avg_err, 'r-', lw=1)
    ax.set_xlabel("Chunk Index", fontsize=11)
    ax.set_ylabel("Position Error (m)", fontsize=11)
    ax.set_title("Error accumulation per chunk", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 2: CUMULATIVE Error Impact per Boundary (how each boundary affects trajectory drift)",
                 fontsize=15, y=0.98)
    out = os.path.join(output_dir, "chunk_stitching_cumulative.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")
    return top_cumul


# ── per-chunk pair comparison ─────────────────────────────────────────────

def plot_chunk_pairs_comparison(gt_xyz, se3_al, owned, top_k, output_dir):
    """
    Compare globally-aligned SE3 vs GT for each chunk.
    Rank chunks by per-chunk ATE (descending), visualize top-K worst.
    Each subplot draws GT (black) and SE3 (red) for that chunk,
    with green lines showing per-frame displacement at sampled frames.
    """
    chunk_stats = []
    for ci, (fs, fe) in enumerate(owned):
        if fe - fs < 2:
            continue
        cg = gt_xyz[fs:fe]
        ce = se3_al[fs:fe]
        dists = np.linalg.norm(cg - ce, axis=1)
        chunk_stats.append(dict(
            chunk_idx=ci, fs=fs, fe=fe,
            ate=np.sqrt(np.mean(dists ** 2)),
            max_err=dists.max(),
        ))

    sorted_chunks = sorted(chunk_stats, key=lambda c: c["ate"], reverse=True)
    top = sorted_chunks[:top_k]
    n = len(top)
    if n == 0:
        return

    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, c in enumerate(top):
        ax = axes[idx]
        fs, fe = c["fs"], c["fe"]
        cg = gt_xyz[fs:fe]
        ce = se3_al[fs:fe]

        ax.plot(cg[:, 0], cg[:, 2], 'k-', lw=2, alpha=0.7, label='GT')
        ax.plot(ce[:, 0], ce[:, 2], 'r-', lw=1.5, alpha=0.8, label='SE3')

        ax.plot(cg[0, 0], cg[0, 2], 'ks', ms=5, zorder=6)
        ax.plot(cg[-1, 0], cg[-1, 2], 'k^', ms=5, zorder=6)
        ax.plot(ce[0, 0], ce[0, 2], 'rs', ms=4, zorder=6)
        ax.plot(ce[-1, 0], ce[-1, 2], 'r^', ms=4, zorder=6)

        nf = fe - fs
        step = max(nf // 8, 1)
        for j in range(0, nf, step):
            ax.plot([cg[j, 0], ce[j, 0]], [cg[j, 2], ce[j, 2]],
                    'g-', lw=0.6, alpha=0.5)

        ax.set_title(
            f"#{idx+1}  Chunk {c['chunk_idx']}  (f{fs}\u2013{fe-1})\n"
            f"ATE={c['ate']:.2f}m  MaxErr={c['max_err']:.2f}m",
            fontsize=7, fontweight='bold')
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=6, loc='best')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Top {n} Worst Chunks: Globally-aligned SE3 vs GT  (ranked by chunk ATE)\n"
        f"(green lines = per-frame error samples)",
        fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, "chunk_pairs_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--se3", required=True)
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--overlap_size", type=int, default=3)
    ap.add_argument("--output_dir", default="results/ate_decomposition")
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gt_poses = read_kitti_poses(args.gt)
    se3_poses = read_tum_poses(args.se3)
    N = min(len(gt_poses), len(se3_poses))
    gt_poses = gt_poses[:N]; se3_poses = se3_poses[:N]
    gt_xyz = np.array([T[:3,3] for T in gt_poses])
    se3_xyz = np.array([T[:3,3] for T in se3_poses])

    windows = compute_windows(N, args.window_size, args.overlap_size)
    owned = owned_ranges(windows)
    stride = windows[1][0] - windows[0][0] if len(windows) > 1 else N
    print(f"Frames={N}, Chunks={len(windows)}, Window={args.window_size}, "
          f"Overlap={args.overlap_size}, Stride={stride}")

    records, se3_al, pfe = analyse_boundaries(
        gt_poses, se3_poses, windows, owned, gt_xyz, se3_xyz)
    print(f"Analysed {len(records)} boundaries")

    top_local = plot_local_errors(records, gt_xyz, se3_al, args.top_k, args.output_dir)
    top_cumul = plot_cumulative_impact(records, gt_xyz, se3_al, pfe, owned, args.top_k, args.output_dir)
    plot_chunk_pairs_comparison(gt_xyz, se3_al, owned, args.top_k, args.output_dir)

    # print summary tables
    print(f"\n{'='*80}")
    print(f"Top {args.top_k} LOCAL stitching errors (rotation)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Boundary':>10} {'Frame':>7} {'RotErr°':>9} {'TransErr(m)':>12}")
    for i, r in enumerate(top_local):
        print(f"{i+1:>4} {r['boundary_idx']:>5}→{r['boundary_idx']+1:<4} "
              f"{r['frame']:>7} {r['local_rot_err']:>9.2f} {r['local_trans_err']:>12.4f}")

    print(f"\n{'='*80}")
    print(f"Top {args.top_k} CUMULATIVE error impacts")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Boundary':>10} {'Frame':>7} {'ErrJump(m)':>11} {'Before(m)':>10} {'After(m)':>10}")
    for i, r in enumerate(top_cumul):
        print(f"{i+1:>4} {r['boundary_idx']:>5}→{r['boundary_idx']+1:<4} "
              f"{r['frame']:>7} {r['cumul_err_jump']:>+11.2f} "
              f"{r['avg_err_before']:>10.2f} {r['avg_err_after']:>10.2f}")


if __name__ == "__main__":
    main()
