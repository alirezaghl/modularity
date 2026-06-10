"""
Neuron-Level Functional Specialization
========================================
Clusters neurons by their z-scored mean activation profiles across tasks,
then validates whether clusters align with biological stream identity (SI).

Two complementary clustering methods are computed per layer:

  1. AGGLOMERATIVE (existing)
     Agglomerative clustering with cosine distance on the (D, n_tasks)
     z-scored activation matrix. Threshold chosen via dendrogram elbow.

  2. DOWNSTREAM MODULARITY (new — adapted from Lange et al. 2022)
     Builds a (D, D) pairwise affinity matrix:
       A[i,j] = max(0, act_z[i] · act_z[j])
     This is the rectified Gram matrix of task-response vectors — an
     approximation of the backward Jacobian (∂output/∂h) that the
     modularity repo uses when a live model is available. Neurons whose
     task-response vectors point in the same direction have high affinity;
     neurons with anti-correlated responses get A[i,j]=0 (disconnected).
     The affinity graph is sparsified (top GN_SPARSIFY_FRAC of edges kept),
     then clustered by maximising the Girvan-Newman modularity score Q via:
       (a) spectral initialisation — recursive leading-eigenvector splitting
       (b) Monte Carlo refinement  — simulated annealing with adaptive
           temperature to maintain target entropy ≈ GN_TARGET_ENTROPY
     Reference: Lange, Rolnick, Kording (2022) TMLR
                "Clustering units in neural networks: upstream vs
                 downstream information"
                https://openreview.net/forum?id=Euf7KofunK

KEY DESIGN DECISIONS:

  Signal: Z-scored mean activation (tuning), not probe importance
    Each neuron is characterized by its mean activation across task datasets,
    z-scored per neuron across tasks. This measures what each neuron RESPONDS
    TO (tuning) — the appropriate signal for biological stream correspondence.
    Probe importance measures what the linear decoder exploits — a property of
    the readout, not the neuron. In neuroscience, tuning and causal importance
    can dissociate; for stream identity questions, tuning is the right concept.
    Z-scoring removes each neuron's baseline and scale so we measure relative
    task preference, not absolute activation level.

  Distance (agglomerative): Cosine on z-scored profiles
    Cosine distance on z-scored profiles measures angular similarity in
    task-preference space. A neuron that fires equally for all tasks has a
    flat z-scored profile and is distant from any specialist neuron.

  Affinity (GN modularity): Rectified dot product
    A[i,j] = max(0, act_z[i] · act_z[j])  — co-tuning affinity.
    Neurons with anti-correlated preferences are disconnected (A=0), not
    negatively connected, because GN modularity requires A >= 0.

  Clustering: Agglomerative (not KMeans) + GN modularity
    KMeans in 6-dimensional activation space always returns k=2 with ARI≈1
    because the data has one dominant axis. Agglomerative clustering with
    cosine distance is data-adaptive, doesn't assume spherical clusters,
    and gives a dendrogram for principled threshold selection.
    GN modularity finds the partition that maximally separates the affinity
    graph — it does not require pre-specifying k.

  UMAP: visualization only
    Clustering happens in the original cosine activation space.
    UMAP 2D projection is purely a sanity check — not used to define clusters.

  SI validation: non-circular
    Clusters derived without any task grouping. After clustering, we ask:
    do data-derived clusters show differential biological SI?
    The Spearman correlation between motion activation score and SI is the
    primary continuous validation — non-circular because clustering doesn't
    use SI or task groupings.

  Motion task grouping: consistent with cross_stream_ablation.py
    MOTION_TASKS = ['intphys', 'ssv2', 'dive48', 'k400']
    APPEAR_TASKS = ['imagenet', 'cifar100']
    K400 is treated as motion throughout the pipeline. The asymmetric 4/2
    split also means z-scoring does not produce exact antisymmetry between
    motion and appearance scores, but rho_appearance is still not reported
    because the ventral validation is covered by cross_stream_ablation.py.

  Null model: permute act_raw then z-score once
    The permutation null shuffles raw activation columns before z-scoring —
    not the already-z-scored matrix — to avoid double-standardization which
    would distort the null distribution.

  Figures (two only):
    1. affinity_si_correlation — Spearman rho(motion_score, SI) across layers
    2. umap_clusters_L{layer}  — scatter | tuning profiles | SI violins
       (shows both agglomerative and GN clusters when both are available)

  Plot data saved for reproduction:
    {model}_plot_data.npz  — all arrays needed to regenerate both figures
                             without re-running feature extraction.

Claim boundary:
  Supported: "neurons show task-tuning profiles that partially align with
    biological stream identity"
  Not supported: "neurons form discrete biological modules"

Output:
    $SCRATCH/DOWNSTREAM/neuron_selectivity_v3/{model}/
        {model}_affinity_si_correlation.png
        {model}_umap_clusters_L{layer}.png   (for each SELECTED_LAYER)
        {model}_plot_data.npz
        {model}_results.npz
        {model}_summary.txt

Reproduce figures from saved data (no feature files needed):
    python neuron_selectivity.py vjepa_16f --replot

Usage:
    python neuron_selectivity.py vjepa_16f
    python neuron_selectivity.py              # all models
    python neuron_selectivity.py vjepa_16f --replot
    python neuron_selectivity.py vjepa_16f --skip-gn   # skip GN clustering
"""

from __future__ import annotations

import sys
import json
import warnings
import numpy as np
from collections import deque
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# =============================================================================
# UMAP — graceful fallback to PCA
# =============================================================================
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. Falling back to PCA for visualization.")

# =============================================================================
# Paths & constants
# =============================================================================

SCRATCH  = Path('/home/ailreza/scratch')
PROJECT  = Path('/home/ailreza/projects/def-shahabkb/ailreza')
OUT_ROOT = SCRATCH / 'DOWNSTREAM/neuron_selectivity_v3'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

D        = 1024
T        = 8
N_LAYERS = 24

AGG_LINKAGE  = 'average'
DIST_THRESH  = 0.4
THRESH_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.10

N_SAMPLE    = 5000
N_BOOTSTRAP = 500
N_PERM      = 100
SEED        = 42

# Downstream modularity (GN) clustering parameters
# Adapted from Lange et al. 2022 (modularity.py / monte_carlo_modularity)
GN_SPARSIFY_FRAC  = 0.20    # keep top 20 % of off-diagonal affinities
GN_MAX_CLUSTERS   = 16      # upper bound on number of communities
MC_STEPS          = 5000    # Monte Carlo annealing steps
GN_TARGET_ENTROPY = 0.15    # target discrete entropy for temperature schedule
GN_EPS            = 1e-15   # numerical floor for adjacency checks

# Task groupings — consistent with cross_stream_ablation.py
# K400 is motion throughout the pipeline.
TASKS        = ['intphys', 'imagenet', 'cifar100', 'k400', 'ssv2', 'dive48']
MOTION_TASKS = ('intphys', 'ssv2', 'dive48', 'k400')   # 4 tasks
APPEAR_TASKS = ('imagenet', 'cifar100')                  # 2 tasks

TASK_LABELS = {
    'intphys' : 'IntPhys2',
    'imagenet': 'ImageNet',
    'cifar100': 'CIFAR-100',
    'k400'    : 'K400',
    'ssv2'    : 'SSv2',
    'dive48'  : 'Dive-48',
}

SELECTED_LAYERS = [0, 4, 8, 12, 16, 20, 23]

# Semantic color system — colors encode meaning, not identity
COLORS = {
    'motion'    : '#1f77b4',   # blue  — dorsal / motion stream
    'appearance': '#d62728',   # red   — ventral / appearance stream
    'neutral'   : '#7f7f7f',   # gray  — generalist neurons
    'sig'       : '#222222',   # black — significance markers
    'model_v2'  : '#A23B72',
    'model_v21' : '#F18F01',
    'videomae'  : '#C73E1D',
}

MODEL_CONFIGS = {
    'vjepa_16f': {
        'label'  : 'V-JEPA2-16f',
        'color'  : COLORS['model_v2'],
        'si_path': SCRATCH / 'clustering_relative/vjepa_16f/specificity_index.json',
        'tasks'  : {
            'intphys': {
                'feat_path': SCRATCH / 'DOWNSTREAM/intphys_features/Main/features.npy',
                'mask_path': SCRATCH / 'DOWNSTREAM/intphys_features/Main/valid_mask.npy',
                'type'     : 'monolithic',
            },
            'imagenet': {'feat_dir': PROJECT / 'imagenet100/vjepa_16f',        'split': 'train', 'type': 'memmap'},
            'cifar100': {'feat_dir': SCRATCH / 'DOWNSTREAM/cifar100_features', 'split': 'train', 'type': 'memmap'},
            'k400'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/k400/vjepa2_16f',   'split': 'val',   'type': 'memmap'},
            'ssv2'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/ssv2/vjepa2_16f',   'split': 'train', 'type': 'memmap'},
            'dive48'  : {'feat_dir': SCRATCH / 'DOWNSTREAM/dive48/vjepa2_16f', 'split': 'train', 'type': 'memmap'},
        },
    },
    'vjepa2_1': {
        'label'  : 'V-JEPA2.1',
        'color'  : COLORS['model_v21'],
        'si_path': SCRATCH / 'clustering_relative/vjepa2_1/specificity_index.json',
        'tasks'  : {
            'intphys': {
                'feat_path': SCRATCH / 'DOWNSTREAM/intphys_vjepa21_vmae_features/Main/vjepa2_1/features.npy',
                'mask_path': SCRATCH / 'DOWNSTREAM/intphys_vjepa21_vmae_features/Main/vjepa2_1/valid_mask.npy',
                'type'     : 'monolithic',
            },
            'imagenet': {'feat_dir': PROJECT / 'imagenet100/vjepa_2_1',                 'split': 'train', 'type': 'memmap'},
            'cifar100': {'feat_dir': SCRATCH / 'DOWNSTREAM/cifar100_features/vjepa2_1', 'split': 'train', 'type': 'memmap'},
            'k400'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/k400/vjepa2_1',              'split': 'val',   'type': 'memmap'},
            'ssv2'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/ssv2/vjepa2_1',              'split': 'train', 'type': 'memmap'},
            'dive48'  : {'feat_dir': SCRATCH / 'DOWNSTREAM/dive48/vjepa2_1',            'split': 'train', 'type': 'memmap'},
        },
    },
    'videomae': {
        'label'  : 'VideoMAE',
        'color'  : COLORS['videomae'],
        'si_path': SCRATCH / 'clustering_relative/videomae/specificity_index.json',
        'tasks'  : {
            'intphys': {
                'feat_path': SCRATCH / 'DOWNSTREAM/intphys_vjepa21_vmae_features/Main/videomae/features.npy',
                'mask_path': SCRATCH / 'DOWNSTREAM/intphys_vjepa21_vmae_features/Main/videomae/valid_mask.npy',
                'type'     : 'monolithic',
            },
            'imagenet': {'feat_dir': PROJECT / 'imagenet100/videomae',                  'split': 'train', 'type': 'memmap'},
            'cifar100': {'feat_dir': SCRATCH / 'DOWNSTREAM/cifar100_features/videomae', 'split': 'train', 'type': 'memmap'},
            'k400'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/k400/videomae',              'split': 'val',   'type': 'memmap'},
            'ssv2'    : {'feat_dir': SCRATCH / 'DOWNSTREAM/ssv2/videomae',              'split': 'train', 'type': 'memmap'},
            'dive48'  : {'feat_dir': SCRATCH / 'DOWNSTREAM/dive48/videomae',            'split': 'train', 'type': 'memmap'},
        },
    },
}


# =============================================================================
# NeurIPS style
# =============================================================================

def set_neurips_style():
    mpl.rcParams.update({
        'font.family'      : 'serif',
        'font.serif'       : ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset' : 'stix',
        'font.size'        : 10,
        'axes.titlesize'   : 11,
        'axes.labelsize'   : 10,
        'axes.linewidth'   : 0.8,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
        'lines.linewidth'  : 2.0,
        'axes.grid'        : True,
        'grid.alpha'       : 0.12,
        'grid.linewidth'   : 0.4,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
        'legend.fontsize'  : 9,
        'legend.frameon'   : False,
        'figure.dpi'       : 150,
        'savefig.dpi'      : 300,
        'savefig.bbox'     : 'tight',
    })

set_neurips_style()


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# Data loading
# =============================================================================

def load_mean_activation(task_cfg, layer_idx, rng, n_sample=N_SAMPLE):
    """
    Load features for one task/layer, subsample, mean-pool over time,
    return mean activation per neuron: shape (D,).

    Subsampling seed is fixed per task (not per layer) so the same clips are
    used at every layer — clean cross-layer comparison.
    """
    if task_cfg['type'] == 'monolithic':
        feat = np.load(str(task_cfg['feat_path']), mmap_mode='r')
        mask = np.load(str(task_cfg['mask_path'])).astype(bool)
        H    = feat[layer_idx, mask].astype(np.float32)

    elif task_cfg['type'] == 'memmap':
        split    = task_cfg['split']
        feat_dir = Path(task_cfg['feat_dir'])
        path     = feat_dir / f'{split}_layer_{layer_idx}_features.npy'
        if not path.exists():
            return None
        H = np.load(str(path), mmap_mode='r').astype(np.float32)
        mask_path = feat_dir / f'{split}_valid_mask.npy'
        if mask_path.exists():
            mask = np.load(str(mask_path)).astype(bool)
            if len(mask) == len(H) and mask.sum() > 0:
                H = H[mask]
    else:
        return None

    if len(H) == 0:
        return None
    if len(H) > n_sample:
        idx = rng.choice(len(H), n_sample, replace=False)
        H   = H[idx]

    cols = H.shape[1]
    if cols % D != 0:
        raise ValueError(f"Feature width {cols} not divisible by D={D}")
    t = cols // D
    H = H.reshape(len(H), t, D).mean(axis=1)
    return H.mean(axis=0)


def build_activation_matrix(task_cfgs, layer_idx, seed=SEED):
    """
    Build (D, n_tasks) z-scored mean activation matrix.

    Returns:
      act_raw    : (D, n_avail) raw mean activations
      act_z      : (D, n_avail) z-scored per neuron across tasks
      avail_tasks: task names in order
      task_mask  : (n_tasks,) bool
    """
    act_raw   = np.full((D, len(TASKS)), np.nan, dtype=np.float32)
    task_mask = np.zeros(len(TASKS), dtype=bool)

    for t_idx, task in enumerate(TASKS):
        if task not in task_cfgs:
            continue
        rng      = np.random.RandomState(seed + t_idx * 7919)
        mean_act = load_mean_activation(task_cfgs[task], layer_idx, rng)
        if mean_act is not None:
            act_raw[:, t_idx] = mean_act
            task_mask[t_idx]  = True

    avail_tasks = [t for t, m in zip(TASKS, task_mask) if m]
    if len(avail_tasks) < 3:
        return None, None, None, None

    act   = act_raw[:, task_mask]
    mu    = act.mean(axis=1, keepdims=True)
    sigma = act.std(axis=1,  keepdims=True) + 1e-8
    act_z = (act - mu) / sigma

    return act, act_z, avail_tasks, task_mask


def load_si(si_path):
    """
    Load SI per layer.
    SI > 0 = ventral-biased (V1/V4/IT).
    SI < 0 = dorsal-biased  (MT/MST).
    Folds T*D layout by peak |SI| per neuron.
    """
    if not Path(si_path).exists():
        print(f"  WARNING: SI file not found: {si_path}")
        return {}

    with open(si_path) as f:
        data = json.load(f)

    si_by_layer = {}
    for k, v in data['SI_per_layer'].items():
        arr = np.array(v, dtype=np.float64)
        if len(arr) != D and len(arr) % D == 0:
            arr      = arr.reshape(-1, D)
            peak_idx = np.abs(arr).argmax(axis=0)
            arr      = arr[peak_idx, np.arange(D)]
        elif len(arr) != D:
            arr = arr[:D]
        si_by_layer[int(k)] = arr

    return si_by_layer


# =============================================================================
# Agglomerative clustering (original method)
# =============================================================================

def cosine_distance_matrix(act_z):
    condensed = pdist(act_z, metric='cosine')
    return np.clip(condensed, 0, 2)


def find_threshold_from_dendrogram(Z, min_clusters=2, max_clusters=8):
    """
    Cut at the largest merge-distance jump. Falls back to DIST_THRESH if
    the elbow gives an out-of-range cluster count.
    """
    merge_dists  = Z[:, 2]
    sorted_dists = np.sort(merge_dists)[::-1]
    diffs        = np.diff(sorted_dists)
    if len(diffs) == 0:
        return DIST_THRESH, merge_dists

    jump_idx    = int(np.argmax(np.abs(diffs)))
    best_thresh = float((sorted_dists[jump_idx] + sorted_dists[jump_idx + 1]) / 2)

    test_labels = fcluster(Z, t=best_thresh, criterion='distance') - 1
    n = len(np.unique(test_labels))
    if n < min_clusters or n > max_clusters:
        best_thresh = DIST_THRESH

    return best_thresh, merge_dists


def cluster_agglomerative(act_z, dist_thresh=DIST_THRESH):
    condensed = cosine_distance_matrix(act_z)
    Z         = linkage(condensed, method=AGG_LINKAGE)
    labels    = fcluster(Z, t=dist_thresh, criterion='distance') - 1
    return labels, Z, len(np.unique(labels))


def threshold_stability_check(act_z, thresholds=THRESH_SWEEP):
    condensed = cosine_distance_matrix(act_z)
    Z         = linkage(condensed, method=AGG_LINKAGE)
    labels_per_thresh = {
        thresh: fcluster(Z, t=thresh, criterion='distance') - 1
        for thresh in thresholds
    }
    ari_adjacents = [
        float(adjusted_rand_score(labels_per_thresh[thresholds[i]],
                                   labels_per_thresh[thresholds[i + 1]]))
        for i in range(len(thresholds) - 1)
    ]
    return (
        {t: len(np.unique(l)) for t, l in labels_per_thresh.items()},
        ari_adjacents,
        labels_per_thresh,
    )


def threshold_stability_with_null(act_z, act_raw, thresholds=THRESH_SWEEP,
                                   n_perm=N_PERM, seed=SEED):
    """
    Permute act_raw columns → z-score once → compare ARI to observed.
    Avoids the double-z-scoring bug from permuting act_z directly.
    """
    _, obs_aris, _ = threshold_stability_check(act_z, thresholds)

    rng      = np.random.RandomState(seed)
    n_pairs  = len(obs_aris)
    null_mat = np.zeros((n_perm, n_pairs))

    for perm_i in range(n_perm):
        act_perm = act_raw.copy()
        for col in range(act_perm.shape[1]):
            act_perm[:, col] = rng.permutation(act_perm[:, col])
        mu         = act_perm.mean(axis=1, keepdims=True)
        sigma      = act_perm.std(axis=1,  keepdims=True) + 1e-8
        act_perm_z = (act_perm - mu) / sigma
        _, perm_aris, _ = threshold_stability_check(act_perm_z, thresholds)
        null_mat[perm_i] = perm_aris

    null_mean = null_mat.mean(axis=0).tolist()
    null_std  = null_mat.std(axis=0).tolist()
    p_vals    = [(null_mat[:, i] >= obs_aris[i]).mean() for i in range(n_pairs)]
    return obs_aris, null_mean, null_std, p_vals


# =============================================================================
# Downstream modularity clustering — Girvan-Newman (Lange et al. 2022)
#
# Adapts the backward-Jacobian-based clustering from the modularity repo to
# the pre-extracted-features setting (no live model available).
#
# Affinity proxy:  A[i,j] = max(0, act_z[i] · act_z[j])
#   act_z[i] ∈ R^(n_tasks) is the neuron's z-scored task-response vector.
#   The dot product measures co-tuning: how similarly two neurons respond
#   across tasks. Rectification ensures A ≥ 0 (required by GN modularity).
#
#   In the modularity repo, affinity comes from the backward Jacobian
#   ∂output/∂h computed by backprop. Here, empirical task-response similarity
#   serves as its surrogate — neurons that "do the same thing downstream"
#   activate similarly across tasks.
# =============================================================================

def downstream_affinity_matrix(act_z: np.ndarray) -> np.ndarray:
    """
    Build (D, D) pairwise co-tuning affinity matrix.

    A[i,j] = max(0, act_z[i] · act_z[j])

    Neurons with aligned task preferences (same stream) get high affinity.
    Anti-correlated neurons (opposite streams) get A[i,j] = 0 so the graph
    does not wire them together — consistent with how the GN algorithm
    treats disconnected components.
    """
    gram = act_z @ act_z.T           # (D, D)  dot products
    gram = np.maximum(gram, 0.0)     # rectify — no negative edges in GN
    np.fill_diagonal(gram, 0.0)      # no self-loops
    return gram.astype(np.float32)


def sparsify_affinity(adj: np.ndarray,
                      fraction: float = GN_SPARSIFY_FRAC) -> np.ndarray:
    """
    Binarize adjacency matrix: keep the top `fraction` of off-diagonal edges.
    Mirrors sparsify() in modularity.py, adapted for numpy.
    """
    adj = adj.copy()
    np.fill_diagonal(adj, 0.0)
    i_idx, j_idx = np.tril_indices(len(adj), k=-1)
    off_diag = adj[i_idx, j_idx]
    nonzero  = off_diag[off_diag > 0]
    if len(nonzero) == 0:
        return np.zeros_like(adj)
    cutoff = np.quantile(nonzero, 1.0 - fraction)
    binary = np.where(adj >= cutoff, 1.0, 0.0).astype(np.float32)
    np.fill_diagonal(binary, 0.0)
    # Ensure symmetry (quantile threshold is applied symmetrically by construction)
    binary = np.maximum(binary, binary.T)
    return binary


def gn_score_np(adj: np.ndarray, labels: np.ndarray) -> float:
    """
    Girvan-Newman modularity Q for hard cluster assignments.

    Q = Σ_k [ e_kk - a_k² ]
    where e_kk = fraction of edges within cluster k,
          a_k  = fraction of all edge-ends in cluster k.

    Port of girvan_newman_sym() from modularity.py, pure numpy.
    """
    total = float(adj.sum())
    if total < GN_EPS:
        return 0.0
    Q = 0.0
    for c in np.unique(labels):
        mask = labels == c
        e_cc = float(adj[np.ix_(mask, mask)].sum()) / total
        a_c  = float(adj[mask].sum())               / total
        Q   += e_cc - a_c ** 2
    return Q


def _softmax_entropy_np(scores: np.ndarray, temperature: float) -> float:
    """Discrete entropy of softmax(scores / temperature)."""
    s = scores / max(temperature, 1e-12)
    s = s - s.max()
    p = np.exp(s);  p /= p.sum()
    return float(-np.sum(p * np.log(p + 1e-300)))


def _entropy_to_temp_np(scores: np.ndarray, target: float,
                         init_t: float = 1.0,
                         eps: float    = 0.01,
                         max_steps: int = 500) -> float:
    """
    Binary-search temperature so that H(softmax(scores/T)) ≈ target.
    Port of entropy_to_temperature() from probability.py, pure numpy.
    """
    log_t = np.log(max(init_t, 1e-6))
    step  = 1.0
    ent   = _softmax_entropy_np(scores, np.exp(log_t))

    for _ in range(max_steps):
        new_log_t = log_t - step if ent > target else log_t + step
        new_ent   = _softmax_entropy_np(scores, np.exp(new_log_t))
        if abs(target - new_ent) < eps:
            break
        if abs(target - ent) < abs(target - new_ent):
            step /= 2
        else:
            log_t, ent = new_log_t, new_ent

    return float(np.clip(np.exp(new_log_t), 1e-12, 1e6))


def spectral_modularity_np(adj: np.ndarray,
                            max_clusters: int = GN_MAX_CLUSTERS) -> np.ndarray:
    """
    Approximate GN-optimal clustering via recursive leading-eigenvector
    splitting of the modularity matrix B = A/sum(A) - a·aᵀ.

    Port of spectral_modularity() from modularity.py, using scipy.linalg.eigh
    instead of torch.svd.

    Returns hard cluster labels (int array, length D).
    """
    from scipy.linalg import eigh

    n     = len(adj)
    total = float(adj.sum())
    if total < GN_EPS:
        return np.zeros(n, dtype=int)

    adj_n = adj / total
    deg   = adj_n.sum(axis=1, keepdims=True)     # (n, 1)
    B     = adj_n - deg @ deg.T                  # modularity matrix

    def _gn_b(lbs: np.ndarray) -> float:
        """GN score using pre-normalised B (avoids repeated adj.sum calls)."""
        Q = 0.0
        for c in np.unique(lbs):
            mask = lbs == c
            Q   += float(B[np.ix_(mask, mask)].sum())
        return Q

    labels     = np.zeros(n, dtype=int)
    best_score = _gn_b(labels)
    next_cid   = 1
    queue      = deque([0])

    while queue and next_cid < max_clusters:
        cid  = queue.popleft()
        mask = labels == cid
        if mask.sum() <= 1:
            continue

        Bsub = B[np.ix_(mask, mask)]
        m    = int(mask.sum())
        idxs = np.where(mask)[0]

        try:
            # Top eigenvector of symmetric Bsub via eigh
            _, eigvecs = eigh(Bsub, subset_by_index=[m - 1, m - 1])
            v = eigvecs[:, 0]
        except Exception:
            continue

        # A split is only meaningful when eigenvector has both signs
        if np.all(v >= 0) or np.all(v <= 0):
            continue

        new_labels = labels.copy()
        new_labels[idxs[v < 0]] = next_cid

        subdiv_score = _gn_b(new_labels)
        if subdiv_score > best_score:
            labels     = new_labels
            best_score = subdiv_score
            queue.extend([cid, next_cid])
            next_cid  += 1

    return labels


def monte_carlo_modularity_np(adj: np.ndarray,
                               labels_init: np.ndarray,
                               steps: int          = MC_STEPS,
                               target_entropy: float = GN_TARGET_ENTROPY,
                               seed: int            = SEED
                               ) -> tuple[np.ndarray, float]:
    """
    Refine cluster assignments via simulated annealing on GN modularity Q.

    Each step:
      1. Pick a random neuron i.
      2. Try reassigning it to each existing cluster (plus one new one).
      3. Sample the new assignment from softmax(Q_candidates / T).
      4. Adapt T to keep discrete entropy ≈ target_entropy.
      5. Track the best (highest Q) assignment seen.

    Port of monte_carlo_modularity() from modularity.py, pure numpy.

    Returns:
      best_labels : (D,) int array, cluster ids re-indexed 0..k-1
      best_score  : corresponding GN Q value
    """
    rng         = np.random.RandomState(seed)
    labels      = labels_init.copy()
    best_labels = labels.copy()
    best_score  = gn_score_np(adj, labels)
    temperature = 1.0

    n_clusters  = int(labels.max()) + 1

    for _ in range(steps):
        idx     = rng.randint(len(labels))
        scores  = np.full(n_clusters + 1, -np.inf)
        used_new = False

        for c in range(n_clusters + 1):
            is_new = (labels == c).sum() == 0
            if is_new:
                if used_new:
                    continue
                used_new = True
            labels[idx] = c
            scores[c]   = gn_score_np(adj, labels)
            if scores[c] > best_score:
                best_score  = scores[c]
                best_labels = labels.copy()

        temperature = _entropy_to_temp_np(scores, target_entropy,
                                           init_t=temperature)
        s = scores / max(temperature, 1e-12)
        s = s - s.max()
        p = np.exp(s);  p /= p.sum()
        choice      = rng.choice(len(scores), p=p)
        labels[idx] = choice
        n_clusters  = max(n_clusters, int(labels.max()) + 1)

    # Re-index cluster ids to 0..k-1 (remove gaps from creation/deletion)
    unique = np.unique(best_labels)
    remap  = {int(old): new for new, old in enumerate(unique)}
    return np.array([remap[int(l)] for l in best_labels], dtype=int), best_score


def cluster_downstream_modularity(act_z: np.ndarray,
                                   sparsify_frac: float  = GN_SPARSIFY_FRAC,
                                   mc_steps: int         = MC_STEPS,
                                   target_entropy: float = GN_TARGET_ENTROPY,
                                   seed: int             = SEED
                                   ) -> tuple[np.ndarray, float, int]:
    """
    Full downstream modularity clustering pipeline:

      1. Build rectified Gram affinity matrix from z-scored task activations
         A[i,j] = max(0, act_z[i] · act_z[j])
      2. Sparsify: keep top `sparsify_frac` of off-diagonal edges (binarize)
      3. Spectral initialization via leading-eigenvector subdivision
      4. Monte Carlo (simulated annealing) refinement

    Returns:
      labels     : (D,) int array of cluster ids
      gn_q       : final Girvan-Newman Q score
      n_clusters : number of clusters found
    """
    adj    = downstream_affinity_matrix(act_z)
    adj_sp = sparsify_affinity(adj, fraction=sparsify_frac)

    if float(adj_sp.sum()) < GN_EPS:
        return np.zeros(len(act_z), dtype=int), 0.0, 1

    labels_spec  = spectral_modularity_np(adj_sp)
    labels_mc, q = monte_carlo_modularity_np(
        adj_sp, labels_spec,
        steps=mc_steps, target_entropy=target_entropy, seed=seed,
    )
    return labels_mc, q, int(labels_mc.max()) + 1


# =============================================================================
# UMAP / PCA projection (visualization only)
# =============================================================================

def compute_2d_coords(act_z, seed=SEED):
    """
    2D embedding of z-scored activation profiles — visualization only.
    Uses UMAP if available, otherwise PCA.
    """
    if HAS_UMAP:
        reducer = umap_lib.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=2,
            random_state=seed,
            metric='cosine',
        )
        return reducer.fit_transform(act_z)
    else:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(act_z)


# =============================================================================
# SI validation
# =============================================================================

def validate_clusters_with_si(labels, si, act_z, avail_tasks):
    """Post-hoc SI validation of data-derived clusters (non-circular)."""
    cluster_ids   = np.unique(labels)
    cluster_stats = {}

    for cid in cluster_ids:
        mask     = labels == cid
        mean_act = act_z[mask].mean(axis=0)
        top_idx  = np.argsort(mean_act)[::-1][:2]

        cluster_stats[int(cid)] = {
            'n_neurons': int(mask.sum()),
            'mean_si'  : float(si[mask].mean()) if si is not None else np.nan,
            'std_si'   : float(si[mask].std())  if si is not None else np.nan,
            'q25_si'   : float(np.percentile(si[mask], 25)) if si is not None else np.nan,
            'q75_si'   : float(np.percentile(si[mask], 75)) if si is not None else np.nan,
            'top_tasks': [avail_tasks[i] for i in top_idx],
            'mean_act' : mean_act.tolist(),
        }

    mwu_results = {}
    if si is not None:
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1:]:
                si1, si2 = si[labels == c1], si[labels == c2]
                if len(si1) > 0 and len(si2) > 0:
                    stat, p = mannwhitneyu(si1, si2, alternative='two-sided')
                    mwu_results[(int(c1), int(c2))] = {
                        'statistic': float(stat), 'p_value': float(p)
                    }

    return cluster_stats, mwu_results


def motion_activation_score(act_z, avail_tasks):
    """
    Per-neuron scalar: mean z-scored activation for motion tasks minus
    appearance tasks. Uses MOTION_TASKS / APPEAR_TASKS — consistent with
    cross_stream_ablation.py. K400 is motion.
    rho_appearance is NOT computed — ventral validation covered by
    cross_stream_ablation.py.
    """
    m_idx = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
    a_idx = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]
    m = act_z[:, m_idx].mean(axis=1) if m_idx else np.zeros(D)
    a = act_z[:, a_idx].mean(axis=1) if a_idx else np.zeros(D)
    return m - a


def bootstrap_spearman_ci(x, y, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for Spearman rho. Joint resampling preserves pairing."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y  = x[valid], y[valid]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = spearmanr(x, y)
    rng    = np.random.RandomState(seed)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx           = rng.choice(len(x), len(x), replace=True)
        boot_rhos[i]  = spearmanr(x[idx], y[idx])[0]
    return (float(rho), float(p),
            float(np.nanpercentile(boot_rhos, 2.5)),
            float(np.nanpercentile(boot_rhos, 97.5)))


# =============================================================================
# Cross-layer stability (logged, not plotted)
# =============================================================================

def cross_layer_stability(labels_by_layer, selected_layers):
    valid = [l for l in selected_layers if l in labels_by_layer]
    layer_pairs, aris = [], []
    for i in range(len(valid) - 1):
        l1, l2 = valid[i], valid[i + 1]
        layer_pairs.append((l1, l2))
        aris.append(float(adjusted_rand_score(labels_by_layer[l1],
                                               labels_by_layer[l2])))
    return layer_pairs, aris


# =============================================================================
# BH-FDR correction
# =============================================================================

def benjamini_hochberg(pvals, alpha=0.05):
    pvals    = np.asarray(pvals, dtype=float)
    rejected = np.zeros(len(pvals), dtype=bool)
    idx_v    = np.where(~np.isnan(pvals))[0]
    if len(idx_v) == 0:
        return rejected
    p, n  = pvals[idx_v], len(idx_v)
    order = np.argsort(p)
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = p[order] <= thresh
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        rej  = np.zeros(n, dtype=bool)
        rej[order[:kmax + 1]] = True
        rejected[idx_v] = rej
    return rejected


# =============================================================================
# Figure 1 — affinity_si_correlation
# =============================================================================

def plot_si_correlation(model_name, label, color, layer_ids,
                         rhos, ps, ci_los, ci_his, out_dir):
    """
    Spearman rho(motion_activation_score, SI) across layers.
    SI > 0 = ventral-biased; SI < 0 = dorsal-biased.
    Expected: rho < 0 in later layers (motion-active neurons are dorsal-biased).

    Title = what is shown. Interpretation belongs in the paper caption.
    """
    layer_ids = np.asarray(layer_ids)
    rhos      = np.asarray(rhos, dtype=float)
    ps        = np.asarray(ps,   dtype=float)
    ci_los    = np.asarray(ci_los, dtype=float)
    ci_his    = np.asarray(ci_his, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    ax.plot(layer_ids, rhos, 'o-', color=color, lw=2.2, ms=4,
            label=label, zorder=3)
    ax.fill_between(layer_ids, ci_los, ci_his, color=color, alpha=0.15)

    sig_mask = ~np.isnan(ps) & (ps < 0.05)
    if sig_mask.any():
        ax.scatter(layer_ids[sig_mask], rhos[sig_mask],
                   s=35, color=COLORS['sig'], zorder=5)

    ax.axhline(0, color='black', lw=0.8, alpha=0.5, ls='--')
    ax.set_xlabel('Layer')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title('Motion tuning vs biological SI')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))

    handles = [
        Line2D([0], [0], color=color, lw=2.2, marker='o', ms=4, label=label),
        Patch(facecolor=color, alpha=0.15, label='95% bootstrap CI'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS['sig'], ms=6, label='p < 0.05'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8)
    despine(ax)
    plt.tight_layout()

    out = out_dir / f'{model_name}_affinity_si_correlation.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


# =============================================================================
# Figure 2 — umap_clusters
# =============================================================================

def _cluster_color(cid, cluster_stats):
    """
    Assign semantic color based on cluster's dominant task type.
    motion-dominant → blue, appearance-dominant → red, generalist → gray.
    """
    top_task = cluster_stats[cid]['top_tasks'][0]
    if top_task in MOTION_TASKS:
        return COLORS['motion']
    if top_task in APPEAR_TASKS:
        return COLORS['appearance']
    return COLORS['neutral']


def plot_umap_clusters(model_name, label, layer_idx, coords_2d,
                        labels_agg, cluster_stats_agg,
                        avail_tasks, si, out_dir,
                        labels_gn=None, cluster_stats_gn=None,
                        gn_score=None, ari_agg_gn=None):
    """
    Multi-panel figure showing both agglomerative and GN clustering results.

    When labels_gn is provided (downstream modularity clustering available):
      Row 1: AGG scatter | AGG tuning profiles | SI violins (AGG)
      Row 2: GN  scatter | GN  tuning profiles | SI violins (GN)
      plus an ARI annotation comparing the two methods.

    When labels_gn is None (original 3-panel layout):
      scatter | tuning profiles | SI violins

    Colors are semantic: blue = motion-dominant, red = appearance-dominant,
    gray = generalist. Task bars colored by stream membership.
    """
    proj_method  = 'UMAP' if HAS_UMAP else 'PCA'
    has_gn       = labels_gn is not None and cluster_stats_gn is not None

    n_rows = 2 if has_gn else 1
    fig    = plt.figure(figsize=(14, 4.2 * n_rows))
    outer  = gridspec.GridSpec(n_rows, 1, hspace=0.55)

    def _draw_row(gs_row, labels, cluster_stats, method_tag):
        unique_cids  = sorted(cluster_stats.keys())
        n_clusters   = len(unique_cids)
        clust_colors = [_cluster_color(cid, cluster_stats) for cid in unique_cids]
        gs_inner     = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_row,
                                                        wspace=0.40)

        # ── Panel: 2D scatter with centroid annotations ────────────────────
        ax = fig.add_subplot(gs_inner[0])
        for idx, cid in enumerate(unique_cids):
            mask = labels == cid
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[clust_colors[idx]], s=6, alpha=0.40, rasterized=True)
            cx = coords_2d[mask, 0].mean()
            cy = coords_2d[mask, 1].mean()
            ax.text(cx, cy, f'C{cid}', fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec=clust_colors[idx], lw=1.2, alpha=0.85))
        ax.set_title(f'Layer {layer_idx}: tuning space [{method_tag}]')
        ax.set_xlabel(f'{proj_method} 1')
        ax.set_ylabel(f'{proj_method} 2')
        ax.set_xticks([]); ax.set_yticks([])
        despine(ax)

        # ── Panel: tuning profiles, bars colored by stream ─────────────────
        ax = fig.add_subplot(gs_inner[1])
        x = np.arange(len(avail_tasks))
        w = 0.8 / max(n_clusters, 1)

        task_stream_colors = [
            COLORS['motion'] if t in MOTION_TASKS else COLORS['appearance']
            for t in avail_tasks
        ]
        for xi, tc in enumerate(task_stream_colors):
            ax.axvspan(xi - 0.45, xi + 0.45 + w * (n_clusters - 1),
                       alpha=0.07, color=tc, zorder=0)

        for idx, cid in enumerate(unique_cids):
            ax.bar(x + idx * w, cluster_stats[cid]['mean_act'], width=w,
                   color=clust_colors[idx], alpha=0.88, label=f'C{cid}')

        ax.axhline(0, color='black', lw=0.7, alpha=0.5)
        ax.set_xticks(x + w * (n_clusters - 1) / 2)
        ax.set_xticklabels([TASK_LABELS[t] for t in avail_tasks],
                            rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean z-scored activation')
        ax.set_title(f'Cluster tuning profiles [{method_tag}]')
        ax.legend(loc='upper right', fontsize=8)

        motion_pos = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
        appear_pos = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]
        if motion_pos:
            ax.text(np.mean(motion_pos) / len(avail_tasks), 1.06,
                    '▼ motion', transform=ax.transAxes,
                    fontsize=8, color=COLORS['motion'], ha='center')
        if appear_pos:
            ax.text(np.mean(appear_pos) / len(avail_tasks), 1.06,
                    '▼ appear.', transform=ax.transAxes,
                    fontsize=8, color=COLORS['appearance'], ha='center')
        despine(ax)

        # ── Panel: SI violin per cluster ────────────────────────────────────
        ax = fig.add_subplot(gs_inner[2])
        if si is not None and n_clusters > 0:
            data_v    = [si[labels == cid] for cid in unique_cids]
            positions = list(range(n_clusters))
            parts = ax.violinplot(data_v, positions=positions,
                                  showmedians=True, showextrema=False)
            for pc, col in zip(parts['bodies'], clust_colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.65)
                pc.set_edgecolor('none')
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.8)

            for pos, d, col in zip(positions, data_v, clust_colors):
                q25, q75 = np.percentile(d, [25, 75])
                ax.plot([pos - 0.08, pos + 0.08], [q25, q25], color=col, lw=1.2)
                ax.plot([pos - 0.08, pos + 0.08], [q75, q75], color=col, lw=1.2)
                ax.plot([pos, pos], [q25, q75], color=col, lw=1.2, alpha=0.6)

            ax.axhline(0, color='black', ls='--', lw=1.2, alpha=0.5)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [f'C{cid}\nSI={cluster_stats[cid]["mean_si"]:+.2f}'
                 for cid in unique_cids],
                fontsize=8
            )
            ax.set_ylabel('SI  (ventral +  /  dorsal −)')
            ax.set_title(f'Biological stream alignment [{method_tag}]')
            despine(ax)
        else:
            ax.text(0.5, 0.5, 'SI not available', ha='center', va='center',
                    transform=ax.transAxes)

    # Draw agglomerative clustering row
    _draw_row(outer[0], labels_agg, cluster_stats_agg, 'Agglomerative')

    # Draw GN clustering row if available
    if has_gn:
        _draw_row(outer[1], labels_gn, cluster_stats_gn, 'GN Modularity')

    # Build suptitle
    title = f'{label} — Neuron specialization  [Layer {layer_idx}]'
    if has_gn and ari_agg_gn is not None:
        q_str = f'  |  GN Q={gn_score:.3f}' if gn_score is not None else ''
        title += f'\nARI(Agg vs GN) = {ari_agg_gn:.3f}{q_str}'
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = out_dir / f'{model_name}_umap_clusters_L{layer_idx:02d}.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


# =============================================================================
# Save / load plot data for figure reproduction
# =============================================================================

def save_plot_data(model_name, out_dir,
                   layer_ids, rhos, ps, ci_los, ci_his, fdr_rejected,
                   plot_data_by_layer):
    """
    Save all arrays needed to regenerate both figures without re-running
    feature extraction.

    Layout:
      layer_ids, rhos, ps, ci_los, ci_his, fdr_rejected   — for Figure 1
      L{layer}_coords_2d, L{layer}_labels_agg, L{layer}_act_z,
      L{layer}_avail_tasks, L{layer}_cluster_json_agg,
      L{layer}_labels_gn, L{layer}_cluster_json_gn,   (if GN was run)
      L{layer}_gn_score, L{layer}_ari_agg_gn,
      L{layer}_si (if available)                           — for Figure 2
    """
    save_dict = dict(
        layer_ids    = np.array(layer_ids),
        rhos         = np.array(rhos, dtype=float),
        ps           = np.array(ps,   dtype=float),
        ci_los       = np.array(ci_los, dtype=float),
        ci_his       = np.array(ci_his, dtype=float),
        fdr_rejected = fdr_rejected,
    )

    for layer_idx, d in plot_data_by_layer.items():
        pfx = f'L{layer_idx:02d}'
        save_dict[f'{pfx}_coords_2d']        = d['coords_2d']
        save_dict[f'{pfx}_labels_agg']       = d['labels_agg']
        save_dict[f'{pfx}_act_z']            = d['act_z']
        save_dict[f'{pfx}_avail_tasks']      = np.array(d['avail_tasks'])
        save_dict[f'{pfx}_cluster_json_agg'] = np.array(
            [json.dumps(d['cluster_stats_agg'])]
        )
        if d.get('labels_gn') is not None:
            save_dict[f'{pfx}_labels_gn']       = d['labels_gn']
            save_dict[f'{pfx}_gn_score']        = np.array([d.get('gn_score', np.nan)])
            save_dict[f'{pfx}_ari_agg_gn']      = np.array([d.get('ari_agg_gn', np.nan)])
            save_dict[f'{pfx}_cluster_json_gn'] = np.array(
                [json.dumps(d['cluster_stats_gn'])]
            )
        if d['si'] is not None:
            save_dict[f'{pfx}_si'] = d['si']

    path = out_dir / f'{model_name}_plot_data.npz'
    np.savez_compressed(str(path), **save_dict)
    print(f"  Saved plot data → {path}")


def load_plot_data(model_name, out_dir):
    """Load saved plot data and reconstruct figure inputs."""
    path = out_dir / f'{model_name}_plot_data.npz'
    if not path.exists():
        raise FileNotFoundError(f"Plot data not found: {path}\n"
                                f"Run without --replot first.")

    npz = np.load(str(path), allow_pickle=True)

    layer_ids    = npz['layer_ids'].tolist()
    rhos         = npz['rhos'].tolist()
    ps           = npz['ps'].tolist()
    ci_los       = npz['ci_los'].tolist()
    ci_his       = npz['ci_his'].tolist()
    fdr_rejected = npz['fdr_rejected']

    saved_layers = sorted({
        int(k.split('_')[0][1:])
        for k in npz.files
        if k.startswith('L') and '_coords_2d' in k
    })

    plot_data_by_layer = {}
    for layer_idx in saved_layers:
        pfx    = f'L{layer_idx:02d}'
        si_key = f'{pfx}_si'

        raw_stats_agg = json.loads(str(npz[f'{pfx}_cluster_json_agg'][0]))

        d = {
            'coords_2d'       : npz[f'{pfx}_coords_2d'],
            'labels_agg'      : npz[f'{pfx}_labels_agg'],
            'act_z'           : npz[f'{pfx}_act_z'],
            'avail_tasks'     : npz[f'{pfx}_avail_tasks'].tolist(),
            'cluster_stats_agg': {int(k): v for k, v in raw_stats_agg.items()},
            'si'              : npz[si_key] if si_key in npz.files else None,
            'labels_gn'       : None,
            'cluster_stats_gn': None,
            'gn_score'        : np.nan,
            'ari_agg_gn'      : np.nan,
        }

        gn_key = f'{pfx}_labels_gn'
        if gn_key in npz.files:
            raw_stats_gn = json.loads(str(npz[f'{pfx}_cluster_json_gn'][0]))
            d['labels_gn']        = npz[gn_key]
            d['cluster_stats_gn'] = {int(k): v for k, v in raw_stats_gn.items()}
            d['gn_score']         = float(npz[f'{pfx}_gn_score'][0])
            d['ari_agg_gn']       = float(npz[f'{pfx}_ari_agg_gn'][0])

        plot_data_by_layer[layer_idx] = d

    return (layer_ids, rhos, ps, ci_los, ci_his,
            fdr_rejected, plot_data_by_layer)


def replot_from_saved(model_name, cfg):
    """Regenerate both figures from saved plot data — no feature files needed."""
    out_dir = OUT_ROOT / model_name
    print(f"  Loading plot data for {model_name}...")
    (layer_ids, rhos, ps, ci_los, ci_his,
     fdr_rejected, plot_data_by_layer) = load_plot_data(model_name, out_dir)

    plot_si_correlation(
        model_name, cfg['label'], cfg['color'],
        np.array(layer_ids), rhos, ps, ci_los, ci_his, out_dir
    )
    for layer_idx, d in sorted(plot_data_by_layer.items()):
        plot_umap_clusters(
            model_name, cfg['label'], layer_idx,
            d['coords_2d'], d['labels_agg'], d['cluster_stats_agg'],
            d['avail_tasks'], d['si'], out_dir,
            labels_gn=d.get('labels_gn'),
            cluster_stats_gn=d.get('cluster_stats_gn'),
            gn_score=d.get('gn_score'),
            ari_agg_gn=d.get('ari_agg_gn'),
        )
    print(f"  Done — figures saved to {out_dir}/")


# =============================================================================
# Main per-model runner
# =============================================================================

def run_model(model_name, cfg, skip_gn: bool = False):
    print(f"\n{'='*60}")
    print(f"  {cfg['label']} — Neuron Functional Specialization")
    if not skip_gn:
        print(f"  Clustering: Agglomerative  +  Downstream GN Modularity")
    print(f"{'='*60}")

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(exist_ok=True)

    si_all = load_si(cfg['si_path'])

    layer_ids             = []
    all_cluster_stats_agg = []
    all_cluster_stats_gn  = []
    labels_by_layer_agg   = {}
    labels_by_layer_gn    = {}
    rhos, ps              = [], []
    ci_los, ci_his        = [], []
    n_clusters_agg_list   = []
    n_clusters_gn_list    = []
    gn_scores_list        = []
    ari_agg_gn_list       = []
    plot_data_by_layer    = {}

    for layer_idx in range(N_LAYERS):
        print(f"  L{layer_idx:02d}", end='', flush=True)

        act, act_z, avail_tasks, _ = build_activation_matrix(
            cfg['tasks'], layer_idx
        )
        if act is None:
            print(" skip")
            continue

        # ── Agglomerative clustering ──────────────────────────────────────
        condensed      = cosine_distance_matrix(act_z)
        Z              = linkage(condensed, method=AGG_LINKAGE)
        best_thresh, _ = find_threshold_from_dendrogram(Z)
        labels_agg, _, n_clust_agg = cluster_agglomerative(act_z,
                                                             dist_thresh=best_thresh)
        n_clusters_agg_list.append(n_clust_agg)

        si = si_all.get(layer_idx)

        if si is not None and len(si) == D:
            cluster_stats_agg, _ = validate_clusters_with_si(
                labels_agg, si, act_z, avail_tasks
            )
        else:
            cluster_stats_agg = {}

        # ── Downstream GN modularity clustering ──────────────────────────
        labels_gn     = None
        cluster_stats_gn = {}
        gn_q          = np.nan
        n_clust_gn    = 0
        ari_agg_gn    = np.nan

        if not skip_gn:
            labels_gn, gn_q, n_clust_gn = cluster_downstream_modularity(act_z)

            if si is not None and len(si) == D:
                cluster_stats_gn, _ = validate_clusters_with_si(
                    labels_gn, si, act_z, avail_tasks
                )

            ari_agg_gn = float(adjusted_rand_score(labels_agg, labels_gn))
            labels_by_layer_gn[layer_idx] = labels_gn

        n_clusters_gn_list.append(n_clust_gn)
        gn_scores_list.append(gn_q)
        ari_agg_gn_list.append(ari_agg_gn)

        # ── Motion × SI Spearman correlation ──────────────────────────────
        mot_score = motion_activation_score(act_z, avail_tasks)
        if si is not None and len(si) == D:
            rho, p, ci_lo, ci_hi = bootstrap_spearman_ci(
                mot_score, si.astype(np.float64)
            )
        else:
            rho = p = ci_lo = ci_hi = np.nan

        rhos.append(rho);     ps.append(p)
        ci_los.append(ci_lo); ci_his.append(ci_hi)
        layer_ids.append(layer_idx)
        all_cluster_stats_agg.append(cluster_stats_agg)
        all_cluster_stats_gn.append(cluster_stats_gn)
        labels_by_layer_agg[layer_idx] = labels_agg

        # ── Per-layer UMAP + save ─────────────────────────────────────────
        if layer_idx in SELECTED_LAYERS:
            coords_2d   = compute_2d_coords(act_z)
            si_for_plot = si if (si is not None and len(si) == D) else None

            plot_umap_clusters(
                model_name, cfg['label'], layer_idx,
                coords_2d, labels_agg, cluster_stats_agg,
                avail_tasks, si_for_plot, out_dir,
                labels_gn=(labels_gn if not skip_gn else None),
                cluster_stats_gn=(cluster_stats_gn if not skip_gn else None),
                gn_score=gn_q,
                ari_agg_gn=(ari_agg_gn if not skip_gn else None),
            )
            plot_data_by_layer[layer_idx] = {
                'coords_2d'        : coords_2d,
                'labels_agg'       : labels_agg,
                'cluster_stats_agg': cluster_stats_agg,
                'labels_gn'        : labels_gn if not skip_gn else None,
                'cluster_stats_gn' : cluster_stats_gn if not skip_gn else None,
                'gn_score'         : gn_q,
                'ari_agg_gn'       : ari_agg_gn,
                'si'               : si_for_plot,
                'act_z'            : act_z,
                'avail_tasks'      : avail_tasks,
            }

        # ── Console summary line ──────────────────────────────────────────
        si_str = ' | '.join(
            f'C{c}:{s["mean_si"]:+.2f}({s["top_tasks"][0]})'
            for c, s in cluster_stats_agg.items()
            if not np.isnan(s['mean_si'])
        ) if cluster_stats_agg else ''
        gn_str = (f' gn_k={n_clust_gn} Q={gn_q:.3f} ARI={ari_agg_gn:.2f}'
                  if not skip_gn else '')
        print(f" agg_k={n_clust_agg} thr={best_thresh:.3f}{gn_str} "
              f"ρm={rho:+.3f} p={p:.3f} [{si_str}]")

    if not layer_ids:
        print(f"  No usable layers for {model_name}")
        return None

    layer_ids    = np.array(layer_ids)
    ps_arr       = np.array(ps, dtype=float)
    fdr_rejected = benjamini_hochberg(ps_arr)
    n_sig_raw    = int(np.nansum(ps_arr < 0.05))
    n_sig_fdr    = int(fdr_rejected.sum())

    # Figure 1
    plot_si_correlation(
        model_name, cfg['label'], cfg['color'],
        layer_ids, rhos, ps, ci_los, ci_his, out_dir
    )

    # Threshold stability null (logged, not plotted)
    ref_layer = min(labels_by_layer_agg.keys(), key=lambda l: abs(l - 12))
    act_ref, act_z_ref, _, _ = build_activation_matrix(cfg['tasks'], ref_layer)
    print(f"\n  Computing threshold stability null (n={N_PERM})...")
    obs_aris, null_mean, null_std, thresh_p = threshold_stability_with_null(
        act_z_ref, act_ref
    )
    print(f"  Observed ARI: {[f'{a:.3f}' for a in obs_aris]}")
    print(f"  Null mean:    {[f'{a:.3f}' for a in null_mean]}")
    print(f"  p-values:     {[f'{p:.3f}' for p in thresh_p]}")

    _, aris_agg = cross_layer_stability(labels_by_layer_agg, SELECTED_LAYERS)
    print(f"  Cross-layer ARI (Agg):  {[f'{a:.3f}' for a in aris_agg]}")
    if not skip_gn and labels_by_layer_gn:
        _, aris_gn = cross_layer_stability(labels_by_layer_gn, SELECTED_LAYERS)
        print(f"  Cross-layer ARI (GN):   {[f'{a:.3f}' for a in aris_gn]}")
        mean_ari = float(np.nanmean(ari_agg_gn_list))
        print(f"  Mean ARI(Agg vs GN):    {mean_ari:.3f}")
    print(f"  Motion SI corr:   {n_sig_raw}/24 layers p<0.05, "
          f"{n_sig_fdr} survive FDR q<0.05")
    print(f"  Clusters/layer (Agg): {n_clusters_agg_list}")
    if not skip_gn:
        print(f"  Clusters/layer  (GN): {n_clusters_gn_list}")

    _print_summary(
        model_name, cfg['label'], layer_ids,
        all_cluster_stats_agg, all_cluster_stats_gn,
        rhos, ps_arr, fdr_rejected,
        gn_scores_list, ari_agg_gn_list, skip_gn,
        out_dir,
    )

    np.savez_compressed(
        str(out_dir / f'{model_name}_results.npz'),
        layer_ids            = layer_ids,
        rhos                 = np.array(rhos),
        ps                   = ps_arr,
        ci_los               = np.array(ci_los),
        ci_his               = np.array(ci_his),
        fdr_rejected         = fdr_rejected,
        n_clusters_agg       = np.array(n_clusters_agg_list),
        n_clusters_gn        = np.array(n_clusters_gn_list),
        gn_scores            = np.array(gn_scores_list, dtype=float),
        ari_agg_gn           = np.array(ari_agg_gn_list, dtype=float),
        cross_layer_aris_agg = np.array(aris_agg),
        thresh_obs_aris      = np.array(obs_aris),
        thresh_null_mean     = np.array(null_mean),
        thresh_null_std      = np.array(null_std),
        thresh_p_vals        = np.array(thresh_p),
    )

    save_plot_data(
        model_name, out_dir,
        layer_ids, rhos, ps_arr, ci_los, ci_his, fdr_rejected,
        plot_data_by_layer,
    )

    return {
        'layer_ids'       : layer_ids,
        'rhos'            : np.array(rhos),
        'ps'              : ps_arr,
        'fdr_rejected'    : fdr_rejected,
        'cross_layer_aris': np.array(aris_agg),
        'n_clusters_agg'  : np.array(n_clusters_agg_list),
        'n_clusters_gn'   : np.array(n_clusters_gn_list),
        'gn_scores'       : np.array(gn_scores_list, dtype=float),
        'ari_agg_gn'      : np.array(ari_agg_gn_list, dtype=float),
    }


def _print_summary(model_name, label, layer_ids,
                   all_cluster_stats_agg, all_cluster_stats_gn,
                   rhos, ps, fdr_rejected,
                   gn_scores, ari_agg_gn, skip_gn,
                   out_dir):
    gn_col = '' if skip_gn else f"{'gn_k':>6} {'gn_Q':>7} {'ARI_ag':>7}"
    lines = [
        '=' * 90,
        f'  {label} — Neuron Specialization Summary',
        f'  Signal: z-scored mean activation (tuning profiles)',
        f'  Method 1: Agglomerative cosine clustering',
        f'  Method 2: Downstream GN modularity clustering'
        + (' [SKIPPED]' if skip_gn else ''),
        f'  Affinity (GN): A[i,j] = max(0, act_z[i]·act_z[j])  — co-tuning',
        f'  Validation: Spearman rho_motion vs SI (non-circular)',
        f'  Task grouping: MOTION={list(MOTION_TASKS)}  APPEAR={list(APPEAR_TASKS)}',
        '=' * 90,
        f"  {'Layer':>6} {'agg_k':>6} {'rho_mot':>8} {'p_mot':>10} {'FDR':>5}  {gn_col}",
        '  ' + '-' * (55 + (0 if skip_gn else 22)),
    ]
    for li, stats_agg, stats_gn, rho, p, fdr, gq, ari in zip(
            layer_ids, all_cluster_stats_agg, all_cluster_stats_gn,
            rhos, ps, fdr_rejected, gn_scores, ari_agg_gn):
        gn_part = (
            f"  {len(stats_gn) if stats_gn else 0:>6} {gq:>7.3f} {ari:>7.3f}"
            if not skip_gn else ''
        )
        lines.append(
            f"  {li:>6} {len(stats_agg) if stats_agg else 0:>6} "
            f"{rho:>8.3f} {p:>10.4f} {'*' if fdr else '':>5}{gn_part}"
        )
    lines += [
        '',
        '  Interpretation:',
        '  - SI > 0 = ventral-biased (V1/V4/IT alignment)',
        '  - SI < 0 = dorsal-biased  (MT/MST alignment)',
        '  - rho_mot < 0: motion-active neurons are dorsal-biased (expected)',
        '  - rho_mot > 0: motion-active neurons are ventral-biased (unexpected)',
        '  - ARI(Agg vs GN): agreement between the two clustering methods',
        '    High ARI → both methods recover the same partition',
        '    Low  ARI → methods disagree; GN may reveal graph-level structure',
        '    that threshold-based agglomerative clustering misses',
        '  - rho_appearance not reported: ventral validation covered by',
        '    cross_stream_ablation.py using probe subspaces.',
        '  - Clustering uses no task grouping — SI validation is non-circular',
        '  - Claim: neurons show task-tuning profiles that partially align',
        '    with biological stream identity',
        '  - Not claimed: neurons form discrete biological modules',
        '',
        '  Reproduce figures without re-running feature extraction:',
        f'    python neuron_selectivity.py {model_name} --replot',
        f'    python neuron_selectivity.py {model_name} --skip-gn  '
        '(agglomerative only)',
    ]
    txt = '\n'.join(lines)
    print(txt)
    with open(str(out_dir / f'{model_name}_summary.txt'), 'w') as f:
        f.write(txt)


# =============================================================================
# main
# =============================================================================

def main():
    replot  = '--replot'  in sys.argv
    skip_gn = '--skip-gn' in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith('--')]

    model_arg = args[0] if args else None
    models    = [model_arg] if model_arg else list(MODEL_CONFIGS.keys())

    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        if replot:
            replot_from_saved(model_name, cfg)
        else:
            run_model(model_name, cfg, skip_gn=skip_gn)

    print(f'\nAll done → {OUT_ROOT}/')


if __name__ == '__main__':
    main()
