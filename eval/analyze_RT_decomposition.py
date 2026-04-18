"""
R vs T Decomposition of Pose-Stitching Error for LoGeR* SE3 on KITTI.

Builds four trajectory variants from per-chunk Sim3-aligned data:
  1. fix_both  — fix R and T at boundaries (full stitch, accumulated drift)
  2. fix_R     — fix only R, leave T errors (shows T's impact)
  3. fix_T     — fix only T, leave R errors (shows R's impact)
  4. floor     — per-chunk independent Sim3 (no stitching error)

Decomposition:
  R_impact = ATE(fix_T) - ATE(floor)   # error from R alone
  T_impact = ATE(fix_R) - ATE(floor)   # error from T alone

Usage:
  python eval/analyze_RT_decomposition.py --seqs 00 01 02 ... \
      --results_dir results/viser_pi3_kitti
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════

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
            if not line or line.startswith("#"):
                continue
            v = list(map(float, line.split()))
            if len(v) >= 8:
                _, tx, ty, tz, qx, qy, qz, qw = v[:8]
                T = np.eye(4); T[:3, 3] = [tx, ty, tz]
                T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
                poses.append(T)
    return poses


# ═══════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════

def align_sim3(gt, es):
    n = gt.shape[0]
    mu_g, mu_e = gt.mean(0), es.mean(0)
    gc, ec = gt - mu_g, es - mu_e
    sig_e = np.sum(ec**2) / n
    cov = gc.T @ ec / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sig_e
    t = mu_g - s * R @ mu_e
    return s * (es @ R.T) + t, s, R, t

def compute_ate(gt, aligned):
    d = np.linalg.norm(gt - aligned, axis=1)
    return np.sqrt(np.mean(d**2)), d

def rot_err_deg(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ═══════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════

def compute_windows(N, ws, ov):
    step = max(ws - ov, 1)
    wins = []
    for s in range(0, N, step):
        e = min(s + ws, N)
        if e - s >= ov or (e == N and s < N):
            wins.append((s, e))
        if e == N:
            break
    return wins

def owned_ranges(wins):
    nc = len(wins)
    if nc <= 1:
        return [(wins[0][0], wins[0][1])]
    stride = wins[1][0] - wins[0][0]
    res = []
    for i in range(nc):
        fs = i * stride
        fe = (i + 1) * stride if i < nc - 1 else wins[-1][1]
        res.append((fs, fe))
    return res


# ═══════════════════════════════════════════════════════════════════
# Core: stitch simulation with selective R / T correction
# ═══════════════════════════════════════════════════════════════════

def per_chunk_align(gt_poses, se3_poses, gt_xyz, se3_xyz, windows):
    """Independently Sim3-align each chunk to GT."""
    chunk_xyz, chunk_R = [], []
    for ci, (ws, we) in enumerate(windows):
        cg, ce = gt_xyz[ws:we], se3_xyz[ws:we]
        if len(cg) < 3:
            chunk_xyz.append(cg.copy())
            chunk_R.append(np.array([p[:3, :3] for p in gt_poses[ws:we]]))
            continue
        al, s, R, t = align_sim3(cg, ce)
        chunk_xyz.append(al)
        chunk_R.append(np.array([R @ se3_poses[j][:3, :3]
                                 for j in range(ws, we)]))
    return chunk_xyz, chunk_R


def stitch_trajectory(chunk_xyz, chunk_R, windows, owned, fix_R, fix_T):
    """
    Stitch per-chunk aligned trajectory with selective corrections.

    fix_R=True  : at each boundary, rotate chunk to match prev rotation
    fix_T=True  : at each boundary, translate chunk to match prev position
    fix_R=False : keep chunk's own rotation (R errors propagate)
    fix_T=False : keep chunk's own position  (T errors propagate)

    Returns: (positions [N,3], rotations [N,3,3])
    """
    N = owned[-1][1]
    nc = len(owned)
    pos = np.zeros((N, 3))
    rot = np.zeros((N, 3, 3))

    ws0 = windows[0][0]
    fs0, fe0 = owned[0]
    for f in range(fs0, fe0):
        pos[f] = chunk_xyz[0][f - ws0]
        rot[f] = chunk_R[0][f - ws0]

    for b in range(nc - 1):
        ws_next = windows[b + 1][0]
        we_prev = windows[b][1]
        fs_next, fe_next = owned[b + 1]

        mid = (ws_next + we_prev) // 2
        idx_next = max(0, min(mid - ws_next,
                              len(chunk_xyz[b + 1]) - 1))

        p_prev = pos[mid]
        R_prev = rot[mid]

        p_next = chunk_xyz[b + 1][idx_next]
        R_next = chunk_R[b + 1][idx_next]

        dR = R_prev @ R_next.T

        if fix_R:
            R_apply = dR
        else:
            R_apply = np.eye(3)

        if fix_T:
            t_apply = p_prev - R_apply @ p_next
        else:
            t_apply = p_next - R_apply @ p_next

        for f in range(fs_next, fe_next):
            idx = max(0, min(f - ws_next,
                             len(chunk_xyz[b + 1]) - 1))
            pos[f] = R_apply @ chunk_xyz[b + 1][idx] + t_apply
            rot[f] = R_apply @ chunk_R[b + 1][idx]

    return pos, rot


# ═══════════════════════════════════════════════════════════════════
# Per-sequence analysis
# ═══════════════════════════════════════════════════════════════════

def analyse_sequence(gt_path, se3_path, ws, ov, seq_name):
    gt_poses = read_kitti_poses(gt_path)
    se3_poses = read_tum_poses(se3_path)
    N = min(len(gt_poses), len(se3_poses))
    gt_poses, se3_poses = gt_poses[:N], se3_poses[:N]
    gt_xyz = np.array([T[:3, 3] for T in gt_poses])
    se3_xyz = np.array([T[:3, 3] for T in se3_poses])

    windows = compute_windows(N, ws, ov)
    owned = owned_ranges(windows)
    nc = len(windows)

    chunk_xyz, chunk_R = per_chunk_align(
        gt_poses, se3_poses, gt_xyz, se3_xyz, windows)

    # ── build four trajectories ──────────────────────────────────
    t_both, _ = stitch_trajectory(chunk_xyz, chunk_R, windows, owned,
                                  fix_R=True,  fix_T=True)
    t_fixR, _ = stitch_trajectory(chunk_xyz, chunk_R, windows, owned,
                                  fix_R=True,  fix_T=False)
    t_fixT, _ = stitch_trajectory(chunk_xyz, chunk_R, windows, owned,
                                  fix_R=False, fix_T=True)

    # floor: per-chunk aligned (no stitching)
    t_floor = np.zeros((N, 3))
    for ci, (fs, fe) in enumerate(owned):
        w = windows[ci][0]
        for f in range(fs, fe):
            idx = max(0, min(f - w, len(chunk_xyz[ci]) - 1))
            t_floor[f] = chunk_xyz[ci][idx]

    # ── globally align + ATE ─────────────────────────────────────
    def do_ate(traj):
        al, _, _, _ = align_sim3(gt_xyz, traj)
        ate, d = compute_ate(gt_xyz, al)
        return ate, d, al

    ate_both, d_both, al_both = do_ate(t_both)
    ate_fixR, d_fixR, al_fixR = do_ate(t_fixR)
    ate_fixT, d_fixT, al_fixT = do_ate(t_fixT)
    ate_floor, d_floor, al_floor = do_ate(t_floor)

    se3_al, _, _, _ = align_sim3(gt_xyz, se3_xyz)
    ate_se3, _ = compute_ate(gt_xyz, se3_al)

    # ── decomposition ────────────────────────────────────────────
    R_impact = max(ate_fixT - ate_floor, 0)
    T_impact = max(ate_fixR - ate_floor, 0)
    both_impact = ate_both - ate_floor
    interaction = both_impact - R_impact - T_impact
    denom = R_impact + T_impact
    R_pct = 100 * R_impact / denom if denom > 1e-6 else 50.0
    T_pct = 100 * T_impact / denom if denom > 1e-6 else 50.0

    # ── per-boundary magnitudes ──────────────────────────────────
    rot_errs, trans_errs = [], []
    for b in range(nc - 1):
        ws_a, we_a = windows[b]
        ws_b = windows[b + 1][0]
        mid = (ws_b + we_a) // 2
        ia = max(0, min(mid - ws_a, len(chunk_xyz[b]) - 1))
        ib = max(0, min(mid - ws_b, len(chunk_xyz[b + 1]) - 1))
        dR = chunk_R[b][ia] @ chunk_R[b + 1][ib].T
        dt = chunk_xyz[b][ia] - chunk_xyz[b + 1][ib]
        rot_errs.append(rot_err_deg(dR))
        trans_errs.append(np.linalg.norm(dt))
    rot_errs = np.array(rot_errs)
    trans_errs = np.array(trans_errs)

    return dict(
        seq=seq_name, N=N, nc=nc,
        ate_se3=ate_se3, ate_both=ate_both,
        ate_fixR=ate_fixR, ate_fixT=ate_fixT, ate_floor=ate_floor,
        R_impact=R_impact, T_impact=T_impact, R_pct=R_pct, T_pct=T_pct,
        interaction=interaction,
        gt_xyz=gt_xyz,
        d_both=d_both, d_fixR=d_fixR, d_fixT=d_fixT, d_floor=d_floor,
        al_both=al_both, al_fixR=al_fixR, al_fixT=al_fixT, al_floor=al_floor,
        rot_errs=rot_errs, trans_errs=trans_errs,
    )


# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════

def plot_sequence(r, outdir):
    """Per-sequence figure: R vs T comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (a) per-frame error curves
    ax = axes[0, 0]
    frames = np.arange(r['N'])
    ax.plot(frames, r['d_both'],  'r-', lw=.5, alpha=.7,
            label=f"fix_both ATE={r['ate_both']:.2f}m")
    ax.plot(frames, r['d_fixT'],  'b-', lw=.5, alpha=.7,
            label=f"fix_T (R err only) ATE={r['ate_fixT']:.2f}m")
    ax.plot(frames, r['d_fixR'],  color='orange', lw=.5, alpha=.7,
            label=f"fix_R (T err only) ATE={r['ate_fixR']:.2f}m")
    ax.plot(frames, r['d_floor'], 'g-', lw=.5, alpha=.7,
            label=f"floor ATE={r['ate_floor']:.2f}m")
    ax.set_xlabel("Frame"); ax.set_ylabel("Position Error (m)")
    ax.set_title("Per-frame error: R-only vs T-only vs Both")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # (b) R vs T bar chart
    ax = axes[0, 1]
    labels = ['Total\nStitch', 'R\nimpact', 'T\nimpact', 'Inter-\naction', 'Floor']
    vals = [r['ate_both'] - r['ate_floor'],
            r['R_impact'], r['T_impact'], r['interaction'],
            r['ate_floor']]
    colors = ['#D32F2F', '#2196F3', '#FF9800', '#9E9E9E', '#4CAF50']
    bars = ax.bar(labels, vals, color=colors, edgecolor='gray', alpha=.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.02,
                f"{v:.2f}m", ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    ax.set_ylabel("ATE contribution (m)")
    ax.set_title(f"R vs T: R={r['R_pct']:.0f}%  T={r['T_pct']:.0f}%")
    ax.grid(axis='y', alpha=.3)

    # (c) trajectory top-view
    ax = axes[1, 0]
    g = r['gt_xyz']
    ax.plot(g[:, 0], g[:, 2], 'k-', lw=1.5, alpha=.5, label='GT')
    ax.plot(r['al_both'][:, 0], r['al_both'][:, 2], 'r-', lw=.7,
            alpha=.6, label='fix_both')
    ax.plot(r['al_fixT'][:, 0], r['al_fixT'][:, 2], 'b-', lw=.7,
            alpha=.6, label='fix_T (R err)')
    ax.plot(r['al_fixR'][:, 0], r['al_fixR'][:, 2], color='orange',
            lw=.7, alpha=.6, label='fix_R (T err)')
    ax.set_aspect('equal'); ax.legend(fontsize=7)
    ax.set_title("Top View (X-Z)"); ax.grid(True, alpha=.3)

    # (d) boundary R & T error magnitudes
    ax = axes[1, 1]
    bi = np.arange(len(r['rot_errs']))
    ax.bar(bi - 0.2, r['rot_errs'], width=0.4, color='#2196F3',
           alpha=.7, label='Rot err (deg)')
    ax2 = ax.twinx()
    ax2.bar(bi + 0.2, r['trans_errs'], width=0.4, color='#FF9800',
            alpha=.7, label='Trans err (m)')
    ax.set_xlabel("Boundary Index"); ax.set_ylabel("Rotation Error (deg)")
    ax2.set_ylabel("Translation Error (m)")
    ax.set_title("Per-boundary R & T discrepancy")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(axis='y', alpha=.3)

    fig.suptitle(f"Seq {r['seq']}: R vs T Decomposition of Pose-Stitching Error"
                 f"\n(R={r['R_pct']:.0f}%  T={r['T_pct']:.0f}%  "
                 f"ATE_SE3={r['ate_se3']:.2f}m)", fontsize=14, y=.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, "RT_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_summary(all_results, outdir):
    """Cross-sequence summary figure."""
    seqs = [r['seq'] for r in all_results]
    R_pcts = [r['R_pct'] for r in all_results]
    T_pcts = [r['T_pct'] for r in all_results]
    R_abs = [r['R_impact'] for r in all_results]
    T_abs = [r['T_impact'] for r in all_results]
    ates = [r['ate_se3'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # (a) stacked percentage bar
    ax = axes[0]
    x = np.arange(len(seqs))
    ax.bar(x, R_pcts, color='#2196F3', label='R (%)')
    ax.bar(x, T_pcts, bottom=R_pcts, color='#FF9800', label='T (%)')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("Percentage (%)"); ax.set_xlabel("Sequence")
    ax.set_title("R vs T share of pose-stitching error")
    ax.legend(); ax.grid(axis='y', alpha=.3)
    for i, (rp, tp) in enumerate(zip(R_pcts, T_pcts)):
        ax.text(i, rp / 2, f"R {rp:.0f}%", ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        ax.text(i, rp + tp / 2, f"T {tp:.0f}%", ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # (b) absolute R & T impact
    ax = axes[1]
    w = 0.35
    ax.bar(x - w/2, R_abs, w, color='#2196F3', label='R impact (m)')
    ax.bar(x + w/2, T_abs, w, color='#FF9800', label='T impact (m)')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("ATE contribution (m)"); ax.set_xlabel("Sequence")
    ax.set_title("Absolute R & T impact on ATE")
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # (c) scatter: R_pct vs ATE
    ax = axes[2]
    sc = ax.scatter(ates, R_pcts, c=x, cmap='tab10', s=100, zorder=5,
                    edgecolor='k', lw=.5)
    for i, s in enumerate(seqs):
        ax.annotate(s, (ates[i], R_pcts[i]), fontsize=9,
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("ATE_SE3 (m)"); ax.set_ylabel("R share (%)")
    ax.set_title("R dominance vs overall ATE")
    ax.axhline(50, color='gray', ls='--', lw=.8, alpha=.5)
    ax.grid(True, alpha=.3)

    fig.suptitle("KITTI 00-10: R vs T Decomposition Summary", fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "RT_decomposition_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=[
        "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"])
    ap.add_argument("--gt_dir", default="data/kitti/dataset/poses")
    ap.add_argument("--se3_dir",
                    default="results/viser_pi3_kitti/LoGeR_star_se3")
    ap.add_argument("--results_dir",
                    default="results/viser_pi3_kitti")
    ap.add_argument("--window_size", type=int, default=32)
    ap.add_argument("--overlap_size", type=int, default=3)
    args = ap.parse_args()

    all_results = []
    for seq in args.seqs:
        gt = os.path.join(args.gt_dir, f"{seq}.txt")
        se3 = os.path.join(args.se3_dir, f"{seq}.txt")
        if not os.path.isfile(gt) or not os.path.isfile(se3):
            print(f"SKIP seq {seq}: missing files")
            continue
        r = analyse_sequence(gt, se3, args.window_size,
                             args.overlap_size, seq)
        outdir = os.path.join(args.results_dir,
                              f"ate_decomposition_{seq}")
        os.makedirs(outdir, exist_ok=True)
        fig_path = plot_sequence(r, outdir)
        print(f"  Saved: {fig_path}")

        # save text
        txt_path = os.path.join(outdir, "RT_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(f"R vs T Decomposition — Seq {seq}\n")
            f.write(f"{'='*55}\n")
            f.write(f"Frames={r['N']}, Chunks={r['nc']}\n\n")
            f.write(f"ATE_SE3 (original):       {r['ate_se3']:.4f} m\n")
            f.write(f"ATE_both (simulated):     {r['ate_both']:.4f} m\n")
            f.write(f"ATE_fix_R (T err only):   {r['ate_fixR']:.4f} m\n")
            f.write(f"ATE_fix_T (R err only):   {r['ate_fixT']:.4f} m\n")
            f.write(f"ATE_floor (per-chunk):    {r['ate_floor']:.4f} m\n\n")
            f.write(f"R impact: {r['R_impact']:.4f} m ({r['R_pct']:.1f}%)\n")
            f.write(f"T impact: {r['T_impact']:.4f} m ({r['T_pct']:.1f}%)\n")
            f.write(f"Interaction: {r['interaction']:.4f} m\n")
            f.write(f"\nBoundary stats:\n")
            f.write(f"  Rot err: mean={r['rot_errs'].mean():.2f}° "
                    f"median={np.median(r['rot_errs']):.2f}° "
                    f"max={r['rot_errs'].max():.2f}°\n")
            f.write(f"  Trans err: mean={r['trans_errs'].mean():.4f}m "
                    f"median={np.median(r['trans_errs']):.4f}m "
                    f"max={r['trans_errs'].max():.4f}m\n")
        print(f"  Saved: {txt_path}")
        all_results.append(r)

    if len(all_results) < 2:
        return

    # ── summary figure ───────────────────────────────────────────
    sum_dir = os.path.join(args.results_dir, "RT_summary")
    os.makedirs(sum_dir, exist_ok=True)
    sum_fig = plot_summary(all_results, sum_dir)
    print(f"\nSummary figure: {sum_fig}")

    # ── summary table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Seq':>4} {'Frames':>6} {'ATE_SE3':>9} {'ATE_both':>9} "
          f"{'R_impact':>9} {'T_impact':>9} {'R%':>6} {'T%':>6} "
          f"{'Dominant':>8}")
    print(f"{'='*80}")
    for r in all_results:
        dom = "R" if r['R_pct'] > 55 else ("T" if r['T_pct'] > 55 else "R≈T")
        print(f"{r['seq']:>4} {r['N']:>6} {r['ate_se3']:>9.2f} "
              f"{r['ate_both']:>9.2f} {r['R_impact']:>9.2f} "
              f"{r['T_impact']:>9.2f} {r['R_pct']:>5.1f}% "
              f"{r['T_pct']:>5.1f}% {dom:>8}")

    sum_txt = os.path.join(sum_dir, "RT_summary.txt")
    with open(sum_txt, 'w') as f:
        f.write("R vs T Decomposition Summary — All Sequences\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'Seq':>4} {'Frames':>6} {'ATE_SE3':>9} {'ATE_both':>9} "
                f"{'R_impact':>9} {'T_impact':>9} {'R%':>6} {'T%':>6} "
                f"{'Dominant':>8}\n")
        f.write(f"{'-'*80}\n")
        for r in all_results:
            dom = "R" if r['R_pct'] > 55 else (
                "T" if r['T_pct'] > 55 else "R≈T")
            f.write(f"{r['seq']:>4} {r['N']:>6} {r['ate_se3']:>9.2f} "
                    f"{r['ate_both']:>9.2f} {r['R_impact']:>9.2f} "
                    f"{r['T_impact']:>9.2f} {r['R_pct']:>5.1f}% "
                    f"{r['T_pct']:>5.1f}% {dom:>8}\n")
    print(f"\nSaved: {sum_txt}")


if __name__ == "__main__":
    main()
