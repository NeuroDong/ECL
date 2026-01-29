import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Ellipse
import torch

fontsize = 40

PALETTE = {
    'class0': "#a1d99b",    # green
    'class1': '#31a354',    # dark green
    'class2': '#ff9896',    # red
    'ellipse': '#3C5488',   # muted violet for ellipses/contours
    'diag': '#6D6D6D',      # gray for diagonal
    'bar_conf': '#7BCBE7',  # greenish for bars (confidence)  123 203 231
    'bar_gap': '#FA7171'    # orangish for gap
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initFun(calibration_paradigm="TopLabel"):
    """
    Layout (symmetric about vertical center):
    Use 2 rows x 6 equal-width columns.
    Top row:
        (a) Source domain spans columns 0-2
        (b) Target domain spans columns 3-5
    Bottom row:
        (c) Uncalibrated spans 0-1
        (d) Soft-ECE spans 2-3
        (e) ECL spans 4-5
    This makes the total width of top row plots equal to the combined width of the three bottom plots.
    """
    set_seed(42)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['hatch.linewidth'] = 2.0
    
    # Turn off interactive mode to prevent automatic display
    plt.ioff()

    # Create two separate figures (two windows):
    # 1) Distribution figure with Source and Target arranged vertically
    # 2) Reliability figure with three reliability diagrams
    fig_dist = plt.figure(figsize=(8, 16))
    ax_source = fig_dist.add_subplot(2, 1, 1)
    ax_target = fig_dist.add_subplot(2, 1, 2)

    fig_rel = plt.figure(figsize=(12, 8))
    # three reliability diagrams arranged horizontally
    ax_rel_uncal = fig_rel.add_subplot(1, 3, 1)
    ax_rel_soft  = fig_rel.add_subplot(1, 3, 2)
    ax_rel_ecl   = fig_rel.add_subplot(1, 3, 3)

    fs_title = 24
    fs_ticks = 18

    ax_source.set_title('(a) Source domain', fontsize=fontsize, y=1.02)
    ax_target.set_title('(b) Target domain', fontsize=fontsize, y=1.02)
    if calibration_paradigm == "TopLabel":
        ax_rel_uncal.set_title('(c) NLL', fontsize=fontsize, y=1.02)
        ax_rel_soft.set_title('(d) Soft-ECE', fontsize=fontsize, y=1.02)
        ax_rel_ecl.set_title('(e) ECL(Ours)', fontsize=fontsize, y=1.02)
    elif calibration_paradigm == "Classwise":
        ax_rel_uncal.set_title('(f) NLL', fontsize=fontsize, y=1.02)
        ax_rel_soft.set_title('(g) Soft-ECE', fontsize=fontsize, y=1.02)
        ax_rel_ecl.set_title('(h) ECL(Ours)', fontsize=fontsize, y=1.02)
    else:
        ax_rel_uncal.set_title('(i) NLL', fontsize=fontsize, y=1.04)
        ax_rel_soft.set_title('(j) Soft-ECE', fontsize=fontsize, y=1.04)
        ax_rel_ecl.set_title('(k) ECL(Ours)', fontsize=fontsize, y=1.04)
    # Tick sizes
    for ax in [ax_source, ax_target]:
        ax.tick_params(axis='both', labelsize=fontsize)

    for ax in [ax_rel_uncal, ax_rel_soft, ax_rel_ecl]:
        ax.tick_params(axis='both', labelsize=fontsize)

    # Reliability axes limits & ticks
    for ax in [ax_rel_uncal, ax_rel_soft, ax_rel_ecl]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])

    # More vertical space to reduce overlap of internal texts (ECE annotations)
    # tighten layouts for both figures
    fig_dist.subplots_adjust(left=0.212, bottom=0.205, right=0.516, top=0.926, wspace=0.155, hspace=0.369)

    if calibration_paradigm == "Canonical":
        fig_rel.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.55, wspace=0.0, hspace=0.2)
    else:
        fig_rel.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.55, wspace=0.116, hspace=0.2)

    return fig_dist, ax_source, ax_target, fig_rel, ax_rel_uncal, ax_rel_soft, ax_rel_ecl

def plt_Reliability_Diagram(ax, bins, bin_confs, bin_accs, ece, test_accuracy=None, bin_counts=None, calib_paradigm="TopLabel"):
    if calib_paradigm == "TopLabel":
        ece_name = "ECE"
    elif calib_paradigm == "Classwise":
        ece_name = "CwECE"
    
    ax.set_axisbelow(True)
    ax.grid(True, which='major', linestyle=(0, (1, 8)), linewidth=2.0, color='gray')
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0,  color=PALETTE['diag'], alpha=1.0)

    ece = 100*ece
    test_accuracy = 100*test_accuracy

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    ax.bar(bin_centers, bin_accs, width=(bins[1]-bins[0]), alpha=1.0, color=PALETTE['bar_conf'], label="Output", edgecolor="black", linewidth=3.0)
    gaps = [bin_accs[i] - bin_confs[i] for i in range(len(bin_accs))]
    ax.bar(bin_centers, gaps, width=(bins[1]-bins[0])*0.6, alpha=0.8, color=PALETTE['bar_gap'], label="Gap", bottom=bin_confs, facecolor='none', hatch='//\\\\', edgecolor=PALETTE['bar_gap'], linewidth=3.0)
    ax.text(0.05, 0.50, "{}={:.1f}\nACC={:.1f}".format(ece_name, ece, test_accuracy), fontsize=fontsize, color='black',ha='left', va='center', bbox=dict(facecolor='#FBE5D6', edgecolor='black', boxstyle='round,pad=0.15'))
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0.0,1.0])
    ax.set_yticks([0.0,1.0])
    ax.set_xlabel("Confidence", fontsize=fontsize)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.xaxis.set_label_coords(0.5, -0.02)
    ax.yaxis.set_label_coords(-0.02, 0.5)
    ax.legend(prop={"family": "Times New Roman","size":fontsize},loc="upper left")
    ax.tick_params(axis='both', labelsize=fontsize)

def plt_canonical_Calibration_Reliability_Diagram(ax, ece, ece_in_anchors, counts_in_anchors, test_accuracy=None):
    """
    Plots the Canonical Calibration Reliability Diagram on a Simplex.
    ece: Scalar Canonical ECE.
    ece_in_anchors: List of errors per anchor/bin.
    counts_in_anchors: List of sample counts per anchor/bin.
    """
    # 1. Deduce N (resolution) from the number of anchors
    # Num_Anchors = (N+1)*(N+2)/2
    L = len(ece_in_anchors)
    # N^2 + 3N + (2-2L) = 0
    # N = (-3 + sqrt(9 - 4(2-2L)))/2 = (-3 + sqrt(1 + 8L))/2
    N = int((-3 + np.sqrt(1 + 8 * L)) / 2)

    # 2. Define Simplex Vertices (Equilateral Triangle)
    # A corresponds to Class 0 (i dimension)
    # B corresponds to Class 1 (j dimension)
    # C corresponds to Class 2 (k dimension)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])

    # 3. Generate Plotting Points (Vertices of subdivision)
    # Matches the recursion order in metrics.py: i (Class 0) from 0 to N
    points = []
    # Loop order must match metrics.py's recursive_gen(N, 3) flattened
    # recursive_gen: for i in 0..N: for j in 0..N-i: k...
    for i in range(N + 1):
        for j in range(N - i + 1):
            k = N - i - j
            
            # Barycentric combination
            # Note: In metrics.py, i is count for dim 0.
            # If we want A to be Class 0 (Prob(C0)=1), then when i=N, p should be A.
            # p = (i*A + j*B + k*C) / N
            if N > 0:
                p = (i * A + j * B + k * C) / N
            else:
                p = C # Degenerate case
            points.append(p)
    points = np.array(points)

    # 4. Triangulate
    tri = mtri.Triangulation(points[:, 0], points[:, 1])

    # 5. Plot
    # We define values at the POINTS (Anchors from metrics.py correspond topologically to these points).
    # Convert vertex errors to face errors for hard binning (block style)
    
    # We must map each anchor (used in canonical_ece) to the triangular face
    # and paint that face with the anchor's ECE value. Reconstruct anchors
    # the same way as in utils.metrics.canonical_ece: anchors = (grid + 1/K)/(N+1).
    def recursive_gen(n, dim):
        if dim == 1:
            return [[n]]
        res = []
        for ii in range(n + 1):
            sub_res = recursive_gen(n - ii, dim - 1)
            for sub in sub_res:
                res.append([ii] + sub)
        return res

    grid = np.array(recursive_gen(N, 3), dtype=np.float32)
    anchors = (grid + 1.0 / 3.0) / (float(N) + 1.0)  # [L, 3]

    # Map anchors (prob vectors) to 2D coordinates inside the big triangle
    anchor_points = np.dot(anchors, np.vstack([A, B, C]))  # [L,2]

    # Find which triangle (face) each anchor falls into and assign face values
    vals = np.array(ece_in_anchors)
    n_faces = tri.triangles.shape[0]
    face_errors = np.zeros(n_faces, dtype=float)

    trifinder = tri.get_trifinder()
    face_idx_for_anchor = trifinder(anchor_points[:, 0], anchor_points[:, 1])
    # If multiple anchors map to same face, average their values
    counts = np.zeros(n_faces, dtype=int)
    for a_idx, f_idx in enumerate(face_idx_for_anchor):
        if f_idx is None or f_idx < 0:
            continue
        face_errors[f_idx] += float(vals[a_idx])
        counts[f_idx] += 1
    nonzero = counts > 0
    face_errors[nonzero] = face_errors[nonzero] / counts[nonzero]

    # Use tripcolor with facecolors, edgecolors='k' for blocks
    # Normalize for colormap if desired, but facecolors accepts raw values and cmap scales them.
    # We prioritize contrasting blocks as in caogao.py
    
    tpc = ax.tripcolor(points[:, 0], points[:, 1], tri.triangles, facecolors=face_errors,
                       cmap='plasma', edgecolors='k', linewidth=0.5)
    
    # Draw boundary
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1.5)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 6. Colorbar
    # Place horizontal colorbar below the triangle
    # We need to access the figure.
    if ax.figure:
        # Create an axis for colorbar? Or use built-in.
        # Adding colorbar to a subplot can resize it.
        # Let's try to add it inside or just below using inset_axes or similar if available,
        # but standard colorbar is safer.
        cbar = ax.figure.colorbar(tpc, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label("Canonical Calibration Error", fontsize=fontsize)
        # Keep label sizing but remove tick marks and tick labels so only colored blocks are shown
        cbar.ax.tick_params(labelsize=fontsize)
        try:
            # Horizontal colorbar: clear x ticks
            cbar.ax.set_xticks([])
            cbar.ax.set_xticklabels([])
        except Exception:
            # Fallback: clear y ticks if orientation differs / matplotlib version
            cbar.ax.set_yticks([])
            cbar.ax.set_yticklabels([])

    # 7. Text Annotation
    ece_pct = 100.0 * ece
    acc_txt = "" if test_accuracy is None else "ACC={:.1f}".format(100.0*test_accuracy)
    
    # Position text at bottom left
    ax.text(0.5, 0.95, "CaECE={:.1f}, {}".format(ece_pct, acc_txt), color='black', ha='center', va='center',
            bbox=dict(facecolor='#FBE5D6', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.9), fontsize=fontsize)
    return tpc

def get_scores(X):
    score0 = np.sin(X[:, 0]) + X[:, 1] - 1.5
    score1 = -np.cos(X[:, 0]) * 1.5 + X[:, 1] * 0.5 + 1.0
    score2 = -X[:, 0] * 0.8 - np.sin(X[:, 1]) * 1.2 - 0.5
    return np.stack([score0, score1, score2], axis=1)

def plot_data(ax, x_data, y_data, is_normal=True, mean=None):
    # improved scatter aesthetics: colorblind-friendly palette, marker edge and alpha
    s = 400
    ax.scatter(x_data[y_data == 0, 0], x_data[y_data == 0, 1],
               c=PALETTE['class0'], s=s, edgecolor='black', linewidth=1.0, alpha=0.9, label='Class 0', zorder=3)
    ax.scatter(x_data[y_data == 1, 0], x_data[y_data == 1, 1],
               c=PALETTE['class1'], s=s, edgecolor='black', linewidth=1.0, alpha=0.9, label='Class 1', marker='s', zorder=3)
    ax.scatter(x_data[y_data == 2, 0], x_data[y_data == 2, 1],
               c=PALETTE['class2'], s=s, edgecolor='black', linewidth=1.0, alpha=0.9, label='Class 2', marker='^', zorder=3)

    # 源/目标域的等高线（椭圆）应以对应均值为中心
    if is_normal and mean is not None:
        ellipse_1_std = Ellipse(xy=mean, width=2, height=2,
                                edgecolor=PALETTE['ellipse'], facecolor='none', linestyle='--', linewidth=1.2, alpha=0.9)
        ax.add_patch(ellipse_1_std)
        ellipse_2_std = Ellipse(xy=mean, width=4, height=4,
                                edgecolor=PALETTE['ellipse'], facecolor='none', linestyle='--', linewidth=1.0, alpha=0.8)
        ax.add_patch(ellipse_2_std)

    # 决策边界（两域相同）
    x1 = np.linspace(-3.5, 4.5, 500)
    x2 = np.linspace(-3.5, 4.5, 500)
    X1, X2 = np.meshgrid(x1, x2)
    grid_X = np.stack([X1.ravel(), X2.ravel()], axis=1)
    scores = get_scores(grid_X)
    score0, score1, score2 = scores[:, 0], scores[:, 1], scores[:, 2]
    
    Z01 = (score0 - score1).reshape(X1.shape)
    Z12 = (score1 - score2).reshape(X1.shape)

    ax.contour(X1, X2, Z01, levels=[0], colors=PALETTE['ellipse'], linestyles='solid', linewidths=2.0, alpha=0.9, zorder=2)
    ax.contour(X1, X2, Z12, levels=[0], colors=PALETTE['ellipse'], linestyles='solid', linewidths=2.0, alpha=0.9, zorder=2)

    ax.set_xlim([-3.5, 4.5])
    ax.set_ylim([-3.5, 4.5])

def labeling_function(X):
    """
    P(Y|X) for 3 classes
    """
    scores = get_scores(X)
    # Use softmax to get probabilities, then sample, or just argmax for deterministic labels
    # For simplicity, we use argmax for clear decision boundaries.
    return np.argmax(scores, axis=1).astype(np.int64)

