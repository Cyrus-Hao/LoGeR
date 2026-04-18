"""
Yaw / Pitch / Roll Decomposition of Rotation Error.

Two complementary analyses:
  A) Per-frame rotation error from globally-aligned SE3 trajectory,
     decomposed into yaw/pitch/roll Euler angles.
  B) Per-boundary: analytical ATE-impact model weighting each axis by
     the geometric lever arm (yaw × horizontal distance, etc.).

Convention (KITTI camera frame):
  R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
  X=right, Y=down, Z=forward
  yaw   → heading change (around Y)
  pitch → vertical tilt  (around X)
  roll  → sideways tilt  (around Z)
"""
import argparse, os
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

def euler_from_rotation(R):
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll).  Returns (yaw, pitch, roll) rad."""
    sp = np.clip(-R[1, 2], -1.0, 1.0)
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        yaw  = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
    else:
        yaw  = np.arctan2(-R[2, 0], R[0, 0])
        roll = 0.0
    return yaw, pitch, roll


# ═══════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════

def compute_windows(N, ws, ov):
    step = max(ws - ov, 1); wins = []
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
    return [(i * stride,
             (i + 1) * stride if i < nc - 1 else wins[-1][1])
            for i in range(nc)]


# ═══════════════════════════════════════════════════════════════════
# Analysis A: per-frame rotation error from actual trajectory
# ═══════════════════════════════════════════════════════════════════

def per_frame_rotation_analysis(gt_poses, se3_poses, gt_xyz, se3_xyz):
    """Globally Sim3-align, then decompose per-frame R error."""
    N = len(gt_poses)
    _, _, R_global, _ = align_sim3(gt_xyz, se3_xyz)

    yaws   = np.zeros(N)
    pitches = np.zeros(N)
    rolls  = np.zeros(N)
    total_deg = np.zeros(N)

    for f in range(N):
        R_gt  = gt_poses[f][:3, :3]
        R_est = R_global @ se3_poses[f][:3, :3]
        R_err = R_gt.T @ R_est
        y, p, r = euler_from_rotation(R_err)
        yaws[f]   = np.degrees(y)
        pitches[f] = np.degrees(p)
        rolls[f]  = np.degrees(r)
        total_deg[f] = rot_err_deg(R_err)

    return yaws, pitches, rolls, total_deg


# ═══════════════════════════════════════════════════════════════════
# Analysis B: per-boundary analytical ATE impact
# ═══════════════════════════════════════════════════════════════════

def per_boundary_impact(gt_poses, se3_poses, gt_xyz, se3_xyz, windows, owned):
    """
    For each boundary, compute the Euler decomposition of the stitching
    rotation discrepancy and estimate each axis's ATE impact using a
    linearized geometric lever-arm model:

      yaw error  × horizontal dist (sqrt(Δx²+Δz²))   → mainly XZ-plane drift
      pitch error × sqrt(Δy²+Δz²)                     → mainly YZ-plane drift
      roll error  × sqrt(Δx²+Δy²)                     → mainly XY-plane drift
    """
    nc = len(windows)
    N = gt_xyz.shape[0]
    _, _, R_global, _ = align_sim3(gt_xyz, se3_xyz)

    # per-chunk Sim3 alignment for boundary discrepancies
    chunk_R = []
    for ci, (ws, we) in enumerate(windows):
        cg, ce = gt_xyz[ws:we], se3_xyz[ws:we]
        if len(cg) < 3:
            chunk_R.append(np.array([p[:3, :3] for p in gt_poses[ws:we]]))
            continue
        _, s, R, t = align_sim3(cg, ce)
        chunk_R.append(np.array([R @ se3_poses[j][:3, :3]
                                 for j in range(ws, we)]))

    b_yaw = np.zeros(nc - 1)
    b_pitch = np.zeros(nc - 1)
    b_roll = np.zeros(nc - 1)
    b_total = np.zeros(nc - 1)

    yaw_impact   = 0.0
    pitch_impact = 0.0
    roll_impact  = 0.0

    stride = windows[1][0] - windows[0][0] if nc > 1 else N

    for b in range(nc - 1):
        ws_a, we_a = windows[b]
        ws_b = windows[b + 1][0]
        mid = (ws_b + we_a) // 2
        ia = max(0, min(mid - ws_a, len(chunk_R[b]) - 1))
        ib = max(0, min(mid - ws_b, len(chunk_R[b + 1]) - 1))

        dR = chunk_R[b][ia] @ chunk_R[b + 1][ib].T
        y, p, r = euler_from_rotation(dR)
        b_yaw[b]   = np.degrees(y)
        b_pitch[b] = np.degrees(p)
        b_roll[b]  = np.degrees(r)
        b_total[b] = rot_err_deg(dR)

        # downstream frames: all owned frames in chunks > b
        # lever arm: displacement from boundary position
        p_boundary = gt_xyz[mid]
        for c in range(b + 1, nc):
            fs, fe = owned[c]
            for f in range(fs, fe):
                dp = gt_xyz[f] - p_boundary
                dx, dy, dz = dp
                # yaw (around Y): affects XZ plane
                h_dist = np.sqrt(dx*dx + dz*dz)
                yaw_impact += abs(y) * h_dist
                # pitch (around X): affects YZ plane
                yz_dist = np.sqrt(dy*dy + dz*dz)
                pitch_impact += abs(p) * yz_dist
                # roll (around Z): affects XY plane
                xy_dist = np.sqrt(dx*dx + dy*dy)
                roll_impact += abs(r) * xy_dist

    total = yaw_impact + pitch_impact + roll_impact
    if total > 1e-8:
        yaw_pct   = 100 * yaw_impact / total
        pitch_pct = 100 * pitch_impact / total
        roll_pct  = 100 * roll_impact / total
    else:
        yaw_pct = pitch_pct = roll_pct = 33.3

    return dict(
        b_yaw=b_yaw, b_pitch=b_pitch, b_roll=b_roll, b_total=b_total,
        yaw_impact=yaw_impact, pitch_impact=pitch_impact,
        roll_impact=roll_impact,
        yaw_pct=yaw_pct, pitch_pct=pitch_pct, roll_pct=roll_pct,
    )


# ═══════════════════════════════════════════════════════════════════
# Per-sequence entry point
# ═══════════════════════════════════════════════════════════════════

def analyse_sequence(gt_path, se3_path, ws, ov, seq):
    gt_poses  = read_kitti_poses(gt_path)
    se3_poses = read_tum_poses(se3_path)
    N = min(len(gt_poses), len(se3_poses))
    gt_poses, se3_poses = gt_poses[:N], se3_poses[:N]
    gt_xyz  = np.array([T[:3, 3] for T in gt_poses])
    se3_xyz = np.array([T[:3, 3] for T in se3_poses])

    windows = compute_windows(N, ws, ov)
    owned   = owned_ranges(windows)

    se3_al, _, _, _ = align_sim3(gt_xyz, se3_xyz)
    ate_se3, d_se3 = compute_ate(gt_xyz, se3_al)

    # Analysis A
    yaws_f, pitches_f, rolls_f, total_f = per_frame_rotation_analysis(
        gt_poses, se3_poses, gt_xyz, se3_xyz)

    # variance decomposition (per-frame)
    y2 = np.mean(yaws_f**2)
    p2 = np.mean(pitches_f**2)
    r2 = np.mean(rolls_f**2)
    s2 = y2 + p2 + r2
    if s2 > 1e-10:
        var_yaw_pct   = 100 * y2 / s2
        var_pitch_pct = 100 * p2 / s2
        var_roll_pct  = 100 * r2 / s2
    else:
        var_yaw_pct = var_pitch_pct = var_roll_pct = 33.3

    # Analysis B
    bnd = per_boundary_impact(gt_poses, se3_poses, gt_xyz, se3_xyz,
                              windows, owned)

    return dict(
        seq=seq, N=N, nc=len(windows), ate_se3=ate_se3,
        yaws_f=yaws_f, pitches_f=pitches_f, rolls_f=rolls_f,
        total_f=total_f, d_se3=d_se3,
        var_yaw_pct=var_yaw_pct, var_pitch_pct=var_pitch_pct,
        var_roll_pct=var_roll_pct,
        gt_xyz=gt_xyz, se3_al=se3_al,
        **bnd,
    )


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

C_YAW, C_PITCH, C_ROLL = '#E53935', '#1E88E5', '#43A047'

def plot_sequence(r, outdir):
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.30)

    frames = np.arange(r['N'])

    # ── row 0: per-frame rotation errors ─────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(frames, np.abs(r['yaws_f']),   '-', color=C_YAW,  lw=.4,
            alpha=.7, label=f"|yaw| RMS={np.sqrt(np.mean(r['yaws_f']**2)):.2f}°")
    ax.plot(frames, np.abs(r['pitches_f']),'-', color=C_PITCH, lw=.4,
            alpha=.7, label=f"|pitch| RMS={np.sqrt(np.mean(r['pitches_f']**2)):.2f}°")
    ax.plot(frames, np.abs(r['rolls_f']),  '-', color=C_ROLL,  lw=.4,
            alpha=.7, label=f"|roll| RMS={np.sqrt(np.mean(r['rolls_f']**2)):.2f}°")
    ax.set_xlabel("Frame"); ax.set_ylabel("Rotation Error (deg)")
    ax.set_title("A: Per-frame |yaw|, |pitch|, |roll| error (globally aligned)")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # ── row 0 right: variance decomposition pie ─────────────────
    ax = fig.add_subplot(gs[0, 2])
    labels = ['Yaw', 'Pitch', 'Roll']
    vals_var = [r['var_yaw_pct'], r['var_pitch_pct'], r['var_roll_pct']]
    ax.pie(vals_var, labels=labels, colors=[C_YAW, C_PITCH, C_ROLL],
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title("A: Rotation error variance share")

    # ── row 1: per-boundary Euler angle bars ─────────────────────
    ax = fig.add_subplot(gs[1, :2])
    bi = np.arange(len(r['b_yaw']))
    ax.bar(bi, np.abs(r['b_yaw']),   color=C_YAW,   alpha=.7, label='|yaw|')
    ax.bar(bi, np.abs(r['b_pitch']), bottom=np.abs(r['b_yaw']),
           color=C_PITCH, alpha=.7, label='|pitch|')
    ax.bar(bi, np.abs(r['b_roll']),
           bottom=np.abs(r['b_yaw']) + np.abs(r['b_pitch']),
           color=C_ROLL,  alpha=.7, label='|roll|')
    ax.set_xlabel("Boundary Index"); ax.set_ylabel("Angle (deg)")
    ax.set_title("B: Per-boundary Euler decomposition of stitching error")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

    # ── row 1 right: analytical ATE impact pie ──────────────────
    ax = fig.add_subplot(gs[1, 2])
    vals_ate = [r['yaw_pct'], r['pitch_pct'], r['roll_pct']]
    ax.pie(vals_ate, labels=labels, colors=[C_YAW, C_PITCH, C_ROLL],
           autopct='%1.1f%%', startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title("B: Analytical ATE impact share\n(angle × lever arm)")

    # ── row 2 left: histogram of per-frame angles ────────────────
    ax = fig.add_subplot(gs[2, 0])
    max_a = max(np.abs(r['yaws_f']).max(),
                np.abs(r['pitches_f']).max(),
                np.abs(r['rolls_f']).max()) * 1.05 + 0.1
    bins = np.linspace(0, max_a, 50)
    ax.hist(np.abs(r['yaws_f']),   bins=bins, color=C_YAW,   alpha=.5,
            label=f"yaw μ={np.abs(r['yaws_f']).mean():.1f}°")
    ax.hist(np.abs(r['pitches_f']),bins=bins, color=C_PITCH, alpha=.5,
            label=f"pitch μ={np.abs(r['pitches_f']).mean():.1f}°")
    ax.hist(np.abs(r['rolls_f']), bins=bins, color=C_ROLL,  alpha=.5,
            label=f"roll μ={np.abs(r['rolls_f']).mean():.1f}°")
    ax.set_xlabel("|Angle| (deg)"); ax.set_ylabel("Count")
    ax.set_title("Per-frame error distribution")
    ax.legend(fontsize=7); ax.grid(axis='y', alpha=.3)

    # ── row 2 mid: trajectory colored by dominant axis ───────────
    ax = fig.add_subplot(gs[2, 1])
    g = r['gt_xyz']
    dom_color = np.zeros((r['N'], 3))
    for f in range(r['N']):
        ay = abs(r['yaws_f'][f])
        ap = abs(r['pitches_f'][f])
        ar = abs(r['rolls_f'][f])
        mx = max(ay, ap, ar, 1e-6)
        dom_color[f] = np.array([ay/mx, 0, 0]) * np.array([0.9, 0.2, 0.2]) + \
                        np.array([ap/mx, 0, 0]) * np.array([0.1, 0.5, 0.9]) + \
                        np.array([ar/mx, 0, 0]) * np.array([0.3, 0.7, 0.3])
        dom_color[f] = np.clip(dom_color[f], 0, 1)
    ax.scatter(g[:, 0], g[:, 2], c=dom_color, s=1, alpha=.8)
    ax.set_aspect('equal')
    ax.set_title("Trajectory: R=yaw, B=pitch, G=roll dominant")
    ax.grid(True, alpha=.3)

    # ── row 2 right: correlation position error vs rotation ──────
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(r['total_f'], r['d_se3'], s=1, alpha=.3, c='gray')
    ax.set_xlabel("Total Rot Error (deg)"); ax.set_ylabel("Position Error (m)")
    ax.set_title("Rotation error vs Position error")
    ax.grid(True, alpha=.3)

    # ── row 3: smoothed per-frame errors per axis ────────────────
    ax = fig.add_subplot(gs[3, :2])
    win = max(r['N'] // 50, 10)
    for arr, c, lab in [(r['yaws_f'], C_YAW, 'yaw'),
                        (r['pitches_f'], C_PITCH, 'pitch'),
                        (r['rolls_f'], C_ROLL, 'roll')]:
        sm = np.convolve(np.abs(arr), np.ones(win)/win, mode='valid')
        ax.plot(sm, color=c, lw=1, alpha=.8, label=lab)
    ax.set_xlabel("Frame"); ax.set_ylabel(f"Smoothed |angle| (win={win})")
    ax.set_title("Smoothed per-frame error per axis")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

    # ── row 3 right: summary bar ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2,
           [r['var_yaw_pct'], r['var_pitch_pct'], r['var_roll_pct']],
           w, color=[C_YAW, C_PITCH, C_ROLL], alpha=.5,
           label='Variance share')
    ax.bar(x + w/2,
           [r['yaw_pct'], r['pitch_pct'], r['roll_pct']],
           w, color=[C_YAW, C_PITCH, C_ROLL], alpha=.85,
           label='ATE impact share')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%"); ax.set_title("Variance vs ATE-impact share")
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

    fig.suptitle(
        f"Seq {r['seq']}: Yaw/Pitch/Roll Decomposition  "
        f"(ATE={r['ate_se3']:.2f}m)\n"
        f"Variance: Y={r['var_yaw_pct']:.0f}% P={r['var_pitch_pct']:.0f}% "
        f"R={r['var_roll_pct']:.0f}%  |  "
        f"ATE-impact: Y={r['yaw_pct']:.0f}% P={r['pitch_pct']:.0f}% "
        f"R={r['roll_pct']:.0f}%",
        fontsize=13, y=.995)
    path = os.path.join(outdir, "RPY_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


def plot_summary(results, outdir):
    seqs = [r['seq'] for r in results]
    n = len(seqs)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(26, 12))

    # (a) ATE-impact stacked bar
    ax = axes[0, 0]
    yp = [r['yaw_pct'] for r in results]
    pp = [r['pitch_pct'] for r in results]
    rp = [r['roll_pct'] for r in results]
    ax.bar(x, yp, color=C_YAW, label='Yaw')
    ax.bar(x, pp, bottom=yp, color=C_PITCH, label='Pitch')
    b2 = [a+b for a, b in zip(yp, pp)]
    ax.bar(x, rp, bottom=b2, color=C_ROLL, label='Roll')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("%"); ax.set_title("ATE-impact share")
    ax.legend(); ax.grid(axis='y', alpha=.3)
    for i in range(n):
        ax.text(i, yp[i]/2, f"{yp[i]:.0f}", ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    # (b) variance stacked bar
    ax = axes[0, 1]
    vy = [r['var_yaw_pct'] for r in results]
    vp = [r['var_pitch_pct'] for r in results]
    vr = [r['var_roll_pct'] for r in results]
    ax.bar(x, vy, color=C_YAW, label='Yaw')
    ax.bar(x, vp, bottom=vy, color=C_PITCH, label='Pitch')
    b3 = [a+b for a, b in zip(vy, vp)]
    ax.bar(x, vr, bottom=b3, color=C_ROLL, label='Roll')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("%"); ax.set_title("Rotation variance share")
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # (c) mean boundary angle per axis
    ax = axes[0, 2]
    w = 0.25
    my = [np.abs(r['b_yaw']).mean() for r in results]
    mp = [np.abs(r['b_pitch']).mean() for r in results]
    mr = [np.abs(r['b_roll']).mean() for r in results]
    ax.bar(x-w, my, w, color=C_YAW,   label='Yaw')
    ax.bar(x,   mp, w, color=C_PITCH, label='Pitch')
    ax.bar(x+w, mr, w, color=C_ROLL,  label='Roll')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("Mean |angle| (deg)"); ax.set_title("Boundary angle errors")
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # (d) RMS per-frame angle per axis
    ax = axes[1, 0]
    ry = [np.sqrt(np.mean(r['yaws_f']**2)) for r in results]
    rp_ = [np.sqrt(np.mean(r['pitches_f']**2)) for r in results]
    rr = [np.sqrt(np.mean(r['rolls_f']**2)) for r in results]
    ax.bar(x-w, ry, w, color=C_YAW,   label='Yaw')
    ax.bar(x,  rp_, w, color=C_PITCH, label='Pitch')
    ax.bar(x+w, rr, w, color=C_ROLL,  label='Roll')
    ax.set_xticks(x); ax.set_xticklabels(seqs)
    ax.set_ylabel("RMS error (deg)"); ax.set_title("Per-frame RMS rotation error")
    ax.legend(); ax.grid(axis='y', alpha=.3)

    # (e) scatter: yaw_pct vs ATE
    ax = axes[1, 1]
    ates = [r['ate_se3'] for r in results]
    ax.scatter(ates, yp, c=C_YAW, s=80, zorder=5, edgecolor='k', lw=.5)
    for i, s in enumerate(seqs):
        ax.annotate(s, (ates[i], yp[i]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel("ATE (m)"); ax.set_ylabel("Yaw ATE-impact %")
    ax.set_title("Yaw dominance vs ATE"); ax.grid(True, alpha=.3)

    # (f) average across all seqs
    ax = axes[1, 2]
    avg_y = np.mean(yp); avg_p = np.mean(pp); avg_r = np.mean(rp)
    ax.bar(['Yaw', 'Pitch', 'Roll'],
           [avg_y, avg_p, avg_r],
           color=[C_YAW, C_PITCH, C_ROLL], edgecolor='gray', alpha=.85)
    for i, v in enumerate([avg_y, avg_p, avg_r]):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12,
                fontweight='bold')
    ax.set_ylabel("Average ATE-impact %")
    ax.set_title("Average across all sequences"); ax.grid(axis='y', alpha=.3)

    fig.suptitle("KITTI 00-10: Yaw / Pitch / Roll Decomposition Summary",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, "RPY_decomposition_summary.png")
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=[
        "00","01","02","03","04","05","06","07","08","09","10"])
    ap.add_argument("--gt_dir",  default="data/kitti/dataset/poses")
    ap.add_argument("--se3_dir", default="results/viser_pi3_kitti/LoGeR_star_se3")
    ap.add_argument("--results_dir", default="results/viser_pi3_kitti")
    ap.add_argument("--window_size",  type=int, default=32)
    ap.add_argument("--overlap_size", type=int, default=3)
    args = ap.parse_args()

    all_r = []
    for seq in args.seqs:
        gt  = os.path.join(args.gt_dir,  f"{seq}.txt")
        se3 = os.path.join(args.se3_dir, f"{seq}.txt")
        if not os.path.isfile(gt) or not os.path.isfile(se3):
            print(f"SKIP {seq}"); continue
        r = analyse_sequence(gt, se3, args.window_size,
                             args.overlap_size, seq)
        od = os.path.join(args.results_dir, f"ate_decomposition_{seq}")
        os.makedirs(od, exist_ok=True)
        fp = plot_sequence(r, od)

        tp = os.path.join(od, "RPY_summary.txt")
        with open(tp, 'w') as f:
            f.write(f"RPY Decomposition — Seq {seq}\n{'='*60}\n")
            f.write(f"Frames={r['N']}, Chunks={r['nc']}, ATE={r['ate_se3']:.4f}m\n\n")
            f.write("--- A: Per-frame rotation variance share ---\n")
            f.write(f"  Yaw:   {r['var_yaw_pct']:.1f}%\n")
            f.write(f"  Pitch: {r['var_pitch_pct']:.1f}%\n")
            f.write(f"  Roll:  {r['var_roll_pct']:.1f}%\n\n")
            f.write("--- B: Analytical ATE-impact share (angle × lever arm) ---\n")
            f.write(f"  Yaw:   {r['yaw_pct']:.1f}%\n")
            f.write(f"  Pitch: {r['pitch_pct']:.1f}%\n")
            f.write(f"  Roll:  {r['roll_pct']:.1f}%\n\n")
            f.write("Boundary angle stats (deg):\n")
            for name, arr in [("yaw", r['b_yaw']),
                              ("pitch", r['b_pitch']),
                              ("roll", r['b_roll'])]:
                a = np.abs(arr)
                f.write(f"  {name:>5}: mean={a.mean():.2f}  "
                        f"median={np.median(a):.2f}  "
                        f"max={a.max():.2f}\n")
            f.write(f"\nPer-frame RMS (deg):\n")
            f.write(f"  yaw:   {np.sqrt(np.mean(r['yaws_f']**2)):.2f}\n")
            f.write(f"  pitch: {np.sqrt(np.mean(r['pitches_f']**2)):.2f}\n")
            f.write(f"  roll:  {np.sqrt(np.mean(r['rolls_f']**2)):.2f}\n")

        print(f"  [{seq}] Var: Y={r['var_yaw_pct']:5.1f}% "
              f"P={r['var_pitch_pct']:5.1f}% R={r['var_roll_pct']:5.1f}%  |  "
              f"ATE: Y={r['yaw_pct']:5.1f}% P={r['pitch_pct']:5.1f}% "
              f"R={r['roll_pct']:5.1f}%")
        all_r.append(r)

    if len(all_r) < 2:
        return

    sd = os.path.join(args.results_dir, "RPY_summary")
    os.makedirs(sd, exist_ok=True)
    sf = plot_summary(all_r, sd)
    print(f"\nSummary: {sf}")

    hdr = (f"{'Seq':>4} {'ATE':>7}  "
           f"{'VarY%':>6} {'VarP%':>6} {'VarR%':>6}  "
           f"{'ImpY%':>6} {'ImpP%':>6} {'ImpR%':>6}  "
           f"{'Dom':>6}")
    sep = '=' * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    lines = [hdr]
    for r in all_r:
        dom = max([('Yaw', r['yaw_pct']),
                   ('Pitch', r['pitch_pct']),
                   ('Roll', r['roll_pct'])], key=lambda x: x[1])[0]
        line = (f"{r['seq']:>4} {r['ate_se3']:>7.2f}  "
                f"{r['var_yaw_pct']:>5.1f}% {r['var_pitch_pct']:>5.1f}% "
                f"{r['var_roll_pct']:>5.1f}%  "
                f"{r['yaw_pct']:>5.1f}% {r['pitch_pct']:>5.1f}% "
                f"{r['roll_pct']:>5.1f}%  {dom:>6}")
        print(line); lines.append(line)

    with open(os.path.join(sd, "RPY_summary.txt"), 'w') as f:
        f.write("RPY Decomposition Summary\n" + sep + "\n")
        for l in lines:
            f.write(l + "\n")
    print(f"\nSaved: {os.path.join(sd, 'RPY_summary.txt')}")


if __name__ == "__main__":
    main()
