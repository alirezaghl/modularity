"""
Downstream Modularity Clustering — aligned with wrongu/modularity repo
========================================================================

Pipeline mirrors the three-layer structure of the original:

  SECTION 1 — ASSOCIATIONS  (associations.py analog)
    Two association graphs per layer:
      adj_down : probe-Jacobian downstream graph
                 A[i,j] = Σ_tasks W_t[:,i]^T W_t[:,j]
                 For a linear probe y=Wh, ∂y/∂h_i = W[:,i], so this is
                 the inner product of output-sensitivity vectors — the
                 direct analog of backward_jac in the original repo.
      adj_tune : task co-tuning graph  (activation-profile / upstream-style)
                 A[i,j] = max(0, act_z[i] · act_z[j])
                 Analog of forward_cov in the original repo.

  SECTION 2 — MODULARITY CLUSTERING  (modularity.py analog)
    spectral initialisation + Monte Carlo refinement of GN Q,
    dead neurons (degree-zero after sparsification) excluded with label -1.

  SECTION 3 — EVALUATION  (eval.py analog)
    Alignment and transfer between the two association graphs,
    permutation null (same optimizer as observed), bootstrap stability,
    SI biological validation.

Paper wording:
  "Following the modularity framework of Lange et al. (2022), we construct
   neuron-neuron association graphs and maximise Newman-Girvan modularity Q.
   For downstream association we use the Jacobian of trained linear probes
   with respect to hidden units; for a linear probe y=Wh this reduces to
   inner products between probe-weight columns. We compare these downstream
   modules to empirical task co-tuning modules and test whether the
   resulting communities align with biological SI."

Output:
    $SCRATCH/DOWNSTREAM/neuron_selectivity_v3/{model}/
        {model}_downstream_clustering.npz
        {model}_downstream_summary.txt

Usage:
    python clustering_downstream.py vjepa_16f
    python clustering_downstream.py              # all models
"""

from __future__ import annotations

import sys
import json
import warnings
import numpy as np
from collections import deque
from pathlib import Path
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings('ignore')


SCRATCH  = Path('/home/ailreza/scratch')
PROJECT  = Path('/home/ailreza/projects/def-shahabkb/ailreza')
OUT_ROOT = SCRATCH / 'DOWNSTREAM/neuron_selectivity_v3'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

D        = 1024
T        = 8
N_LAYERS = 24

N_SAMPLE = 5000
SEED     = 42

TASKS        = ['intphys', 'imagenet', 'cifar100', 'k400', 'ssv2', 'dive48']
MOTION_TASKS = ('intphys', 'ssv2', 'dive48')
APPEAR_TASKS = ('imagenet', 'cifar100', 'k400')

SELECTED_LAYERS = [0, 4, 8, 12, 16, 20, 23]

# GN modularity parameters
GN_SPARSIFY_FRAC  = 0.20
GN_MAX_CLUSTERS   = 16
MC_STEPS          = 5000
GN_TARGET_ENTROPY = 0.15
GN_EPS            = 1e-15

# Robustness parameters
N_PERM            = 100
N_BOOT            = 100
BOOT_FRAC         = 0.80
N_SHUFFLE_ALIGN   = 1000
GENERALIST_THRESH = 0.20


MODEL_CONFIGS = {
    'vjepa_16f': {
        'label'  : 'V-JEPA2-16f',
        'si_path': SCRATCH / 'clustering_relative/vjepa_16f/specificity_index.json',
        # Probe weight paths: user fills these in.
        # Expected format per task: numpy array W of shape [n_classes, D] or [D].
        # 'probe_weights': {
        #     'intphys' : SCRATCH / 'probes/vjepa_16f/intphys_weights.npy',
        #     'imagenet': SCRATCH / 'probes/vjepa_16f/imagenet_weights.npy',
        #     ...
        # },
        'probe_weights': {},   # populate with paths before running
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
        'si_path': SCRATCH / 'clustering_relative/vjepa2_1/specificity_index.json',
        'probe_weights': {},
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
        'si_path': SCRATCH / 'clustering_relative/videomae/specificity_index.json',
        'probe_weights': {},
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
# Data loading
# =============================================================================

def load_mean_activation(task_cfg, layer_idx, rng, n_sample=N_SAMPLE):
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
        return None, None, None

    act   = act_raw[:, task_mask]
    mu    = act.mean(axis=1, keepdims=True)
    sigma = act.std(axis=1,  keepdims=True) + 1e-8
    act_z = (act - mu) / sigma
    return act, act_z, avail_tasks


def load_probe_weights(probe_weight_paths: dict, layer_idx: int,
                       avail_tasks: list[str]) -> dict:
    """
    Load linear probe weight matrices for available tasks at a given layer.

    Expected file format for each task: a numpy array of shape [n_classes, D]
    or [D] (binary/logistic). Files may be layer-specific:
      path/task_layer_{layer_idx}_weights.npy
    or a single file containing all layers:
      path/task_weights.npy  with shape [n_layers, n_classes, D]

    Returns dict task -> W array [n_classes, D], skips tasks with missing files.
    """
    weights = {}
    for task in avail_tasks:
        if task not in probe_weight_paths:
            continue
        p = Path(probe_weight_paths[task])

        # Try layer-specific file first
        layer_p = p.parent / f'{p.stem}_layer_{layer_idx}{p.suffix}'
        if layer_p.exists():
            W = np.load(str(layer_p))
        elif p.exists():
            W = np.load(str(p))
            # If 3-D, index the layer dimension
            if W.ndim == 3:
                W = W[layer_idx]
        else:
            continue

        W = W.astype(np.float32)
        if W.ndim == 1:
            W = W[None, :]          # [1, D]
        # Ensure shape is [n_classes, D]
        if W.shape[0] == D and W.shape[-1] != D:
            W = W.T
        if W.shape[-1] != D:
            print(f"  WARNING: probe W for {task} layer {layer_idx} "
                  f"has unexpected shape {W.shape}, skipping")
            continue
        weights[task] = W

    return weights


def load_si(si_path):
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
# SECTION 1 — ASSOCIATIONS
# Two graphs: probe-Jacobian downstream (primary) + task co-tuning (secondary)
# =============================================================================

def downstream_probe_jacobian_affinity(probe_weights_by_task: dict,
                                        avail_tasks: list[str]) -> np.ndarray:
    """
    Original-style downstream association using linear-probe Jacobians.

    For a linear probe y = W h,  ∂y/∂h_i = W[:, i].
    Neuron-neuron association = inner product of output-sensitivity vectors,
    summed across downstream tasks:

        A[i,j] = Σ_tasks  W_t[:,i]^T  W_t[:,j]  =  Σ_tasks (W_t.T @ W_t)[i,j]

    This is the direct analog of backward_jac in associations.py of the
    original repo: G_i · G_j where G_i = ∂y/∂h_i.

    Returns (D, D) float32, nonneg, zero diagonal.
    """
    A = np.zeros((D, D), dtype=np.float64)
    n_tasks_used = 0
    for task in avail_tasks:
        if task not in probe_weights_by_task:
            continue
        W = probe_weights_by_task[task].astype(np.float64)  # [C, D]
        A += W.T @ W
        n_tasks_used += 1

    if n_tasks_used == 0:
        return None

    A = A.astype(np.float32)
    np.maximum(A, 0.0, out=A)
    np.fill_diagonal(A, 0.0)
    return A


def task_cotuning_affinity_matrix(act_z: np.ndarray) -> np.ndarray:
    """
    Empirical task co-tuning association (activation-profile / upstream-style).
    Analog of forward_cov in the original repo.

        A[i,j] = max(0, act_z[i] · act_z[j])

    Used as the second association graph for comparison against the
    probe-Jacobian downstream graph.
    """
    gram = (act_z @ act_z.T).astype(np.float32)
    np.maximum(gram, 0.0, out=gram)
    np.fill_diagonal(gram, 0.0)
    return gram


# =============================================================================
# SECTION 2 — MODULARITY CLUSTERING
# sparsify → spectral init → Monte Carlo refinement → handle dead neurons
# =============================================================================

def sparsify_affinity(adj: np.ndarray,
                      fraction: float = GN_SPARSIFY_FRAC) -> np.ndarray:
    """
    Keep the top `fraction` of all off-diagonal entries, weighted (not binary).

    Mirrors sparsify() in modularity.py: accounts for already-zero entries
    before thresholding so that `fraction` is relative to all edges, not just
    non-zero ones. Keeps weighted values (not binarized) as in the original.
    """
    adj = adj.copy().astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    i_idx, j_idx = np.tril_indices(len(adj), k=-1)
    vals    = adj[i_idx, j_idx]
    nonzero = vals[vals > 0]
    if nonzero.size == 0:
        return np.zeros_like(adj)

    # fraction of all off-diagonal entries that are already zero
    zero_frac = 1.0 - nonzero.size / vals.size
    if fraction <= zero_frac:
        return np.zeros_like(adj)

    # fraction to keep among nonzero entries so that overall kept = fraction
    effective_keep = (fraction - zero_frac) / (1.0 - zero_frac)
    effective_keep = float(np.clip(effective_keep, 0.0, 1.0))

    cutoff = np.quantile(nonzero, 1.0 - effective_keep)
    out    = np.where(adj >= cutoff, adj, 0.0).astype(np.float32)  # weighted
    np.fill_diagonal(out, 0.0)
    return np.maximum(out, out.T)   # ensure symmetry


def gn_score(adj: np.ndarray, labels: np.ndarray) -> float:
    """Q = Σ_k [ e_kk - a_k² ]. Port of girvan_newman_sym() from modularity.py."""
    total = float(adj.sum())
    if total < GN_EPS:
        return 0.0
    Q = 0.0
    for c in np.unique(labels[labels >= 0]):
        mask = labels == c
        e_cc = float(adj[np.ix_(mask, mask)].sum()) / total
        a_c  = float(adj[mask].sum()) / total
        Q   += e_cc - a_c ** 2
    return Q


def _entropy_to_temp(scores: np.ndarray, target: float,
                      init_t: float = 1.0, eps: float = 0.01,
                      max_steps: int = 500) -> float:
    """Port of entropy_to_temperature() from probability.py."""
    log_t     = np.log(max(init_t, 1e-6))
    new_log_t = log_t
    step      = 1.0

    def _ent(lt):
        s = scores / max(np.exp(lt), 1e-12)
        s = s - s.max()
        p = np.exp(s); p /= p.sum()
        return float(-np.sum(p * np.log(p + 1e-300)))

    ent = _ent(log_t)
    for _ in range(max_steps):
        new_log_t = log_t - step if ent > target else log_t + step
        new_ent   = _ent(new_log_t)
        if abs(target - new_ent) < eps:
            break
        if abs(target - ent) < abs(target - new_ent):
            step /= 2
        else:
            log_t, ent = new_log_t, new_ent
    return float(np.clip(np.exp(new_log_t), 1e-12, 1e6))


def spectral_modularity(adj: np.ndarray,
                         max_clusters: int = GN_MAX_CLUSTERS) -> np.ndarray:
    """
    Recursive leading-eigenvector splitting of B = A/m - a·aᵀ.
    Port of spectral_modularity() from modularity.py.
    """
    n     = len(adj)
    total = float(adj.sum())
    if total < GN_EPS:
        return np.zeros(n, dtype=int)

    adj_n = adj / total
    deg   = adj_n.sum(axis=1, keepdims=True)
    B     = adj_n - deg @ deg.T

    def _q_from_B(lbs):
        Q = 0.0
        for c in np.unique(lbs):
            mask = lbs == c
            Q   += float(B[np.ix_(mask, mask)].sum())
        return Q

    labels     = np.zeros(n, dtype=int)
    best_score = _q_from_B(labels)
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
            _, eigvecs = eigh(Bsub, subset_by_index=[m - 1, m - 1])
            v = eigvecs[:, 0]
        except Exception:
            continue
        if np.all(v >= 0) or np.all(v <= 0):
            continue
        new_labels = labels.copy()
        new_labels[idxs[v < 0]] = next_cid
        new_score = _q_from_B(new_labels)
        if new_score > best_score:
            labels     = new_labels
            best_score = new_score
            queue.extend([cid, next_cid])
            next_cid  += 1

    return labels


def monte_carlo_modularity(adj: np.ndarray,
                            labels_init: np.ndarray,
                            steps: int           = MC_STEPS,
                            target_entropy: float = GN_TARGET_ENTROPY,
                            seed: int             = SEED
                            ) -> tuple[np.ndarray, float]:
    """
    Simulated annealing on GN Q with vectorized delta-Q cache.
    Port of monte_carlo_modularity() from modularity.py.
    """
    rng   = np.random.RandomState(seed)
    n     = len(adj)
    total = float(adj.sum())
    if total < GN_EPS:
        return np.zeros(n, dtype=int), 0.0

    labels      = labels_init.copy().astype(int)
    n_clusters  = int(labels.max()) + 1
    deg         = adj.sum(axis=1)

    cluster_deg = np.zeros(n_clusters + GN_MAX_CLUSTERS, dtype=np.float64)
    for k in range(n_clusters):
        cluster_deg[k] = deg[labels == k].sum()

    max_k    = n_clusters + GN_MAX_CLUSTERS
    row_sums = np.zeros((n, max_k), dtype=np.float64)
    for k in range(n_clusters):
        row_sums[:, k] = adj[:, labels == k].sum(axis=1)

    current_q = 0.0
    for k in range(n_clusters):
        mask       = labels == k
        e_kk       = float(adj[np.ix_(mask, mask)].sum()) / total
        a_k        = cluster_deg[k] / total
        current_q += e_kk - a_k ** 2

    best_q      = current_q
    best_labels = labels.copy()
    temperature = 1.0

    for _ in range(steps):
        i      = rng.randint(n)
        c_old  = int(labels[i])
        ri     = adj[i]
        deg_i  = float(deg[i])
        self_w = float(adj[i, i])

        candidates = list(range(n_clusters)) + [n_clusters]
        scores     = np.full(len(candidates), -np.inf)

        for ci_idx, c_new in enumerate(candidates):
            if c_new == c_old:
                scores[ci_idx] = current_q
                continue
            rs_new = 0.0 if c_new == n_clusters else float(row_sums[i, c_new])
            dg_new = 0.0 if c_new == n_clusters else float(cluster_deg[c_new])
            rs_old = float(row_sums[i, c_old])
            dg_old = float(cluster_deg[c_old])
            delta  = (2.0 * (rs_new - rs_old - self_w) / total
                      - 2.0 * deg_i * (dg_new - dg_old) / (total ** 2))
            scores[ci_idx] = current_q + delta

        best_ci = int(np.argmax(scores))
        if scores[best_ci] > best_q:
            best_q      = scores[best_ci]
            tmp         = labels.copy()
            tmp[i]      = candidates[best_ci]
            best_labels = tmp

        temperature = _entropy_to_temp(scores, target_entropy, init_t=temperature)
        s = scores / max(temperature, 1e-12)
        s = s - s.max()
        p = np.exp(s); p /= p.sum()
        choice = candidates[rng.choice(len(candidates), p=p)]

        if choice != c_old:
            is_new = (choice == n_clusters)
            if is_new:
                cluster_deg[choice] = 0.0
                n_clusters += 1
                if n_clusters >= max_k:
                    extra       = GN_MAX_CLUSTERS
                    cluster_deg = np.concatenate([cluster_deg, np.zeros(extra)])
                    row_sums    = np.concatenate([row_sums, np.zeros((n, extra))],
                                                  axis=1)
                    max_k      += extra
            labels[i] = choice
            row_sums[:, choice] += ri
            row_sums[:, c_old]  -= ri
            cluster_deg[choice] += deg_i
            cluster_deg[c_old]  -= deg_i
            current_q = 0.0
            for k in range(n_clusters):
                mk = labels == k
                if not mk.any():
                    continue
                e_kk       = float(row_sums[mk, k].sum()) / total
                a_k        = cluster_deg[k] / total
                current_q += e_kk - a_k ** 2

    unique = np.unique(best_labels)
    remap  = {int(old): new for new, old in enumerate(unique)}
    return np.array([remap[int(l)] for l in best_labels], dtype=int), float(best_q)


def cluster_modularity_from_adj(adj_sp: np.ndarray,
                                 mc_steps: int = MC_STEPS,
                                 seed: int     = SEED
                                 ) -> tuple[np.ndarray, float, int]:
    """
    Run spectral init + Monte Carlo refinement on a pre-sparsified adjacency.
    Clean separation of clustering from association construction.
    """
    if float(adj_sp.sum()) < GN_EPS:
        return np.zeros(len(adj_sp), dtype=int), 0.0, 1
    labels_spec = spectral_modularity(adj_sp)
    labels_mc, q = monte_carlo_modularity(adj_sp, labels_spec,
                                           steps=mc_steps, seed=seed)
    return labels_mc, q, int(labels_mc.max()) + 1


def cluster_modularity_alive(adj_sp: np.ndarray,
                              mc_steps: int = MC_STEPS,
                              seed: int     = SEED
                              ) -> tuple[np.ndarray, float, int, np.ndarray]:
    """
    Cluster only non-isolated neurons. Dead neurons (degree-zero) get label -1.

    Mirrors the original repo's handling of 'dead' units: spectral_modularity()
    and monte_carlo_modularity() in modularity.py both explicitly exclude units
    whose degree is below ADJACENCY_EPS before clustering.

    Returns:
      labels_full : (D,) — cluster id or -1 for dead neurons
      q           : GN Q of the alive subgraph
      k           : number of clusters found
      alive       : (D,) bool mask of non-dead neurons
    """
    deg   = adj_sp.sum(axis=1)
    alive = deg > GN_EPS
    labels_full = np.full(len(adj_sp), -1, dtype=int)

    if alive.sum() < 2:
        return labels_full, 0.0, 0, alive

    adj_alive              = adj_sp[np.ix_(alive, alive)]
    labels_alive, q, k     = cluster_modularity_from_adj(adj_alive, mc_steps, seed)
    labels_full[alive]     = labels_alive
    return labels_full, q, k, alive


# =============================================================================
# SECTION 3 — EVALUATION
# alignment, transfer, robustness, SI validation — mirrors eval.py
# =============================================================================

def greedy_alignment_score(labels_a: np.ndarray,
                            labels_b: np.ndarray) -> float:
    """
    Greedy cluster matching maximizing overlap.
    Port of alignment_score() from modularity.py.
    Ignores dead neurons (label == -1).
    """
    valid    = (labels_a >= 0) & (labels_b >= 0)
    la, lb   = labels_a[valid], labels_b[valid]
    ids_a    = np.unique(la)
    ids_b    = np.unique(lb)
    k        = max(len(ids_a), len(ids_b))
    overlap  = np.zeros((k, k), dtype=np.float64)
    for ia, ca in enumerate(ids_a):
        for ib, cb in enumerate(ids_b):
            overlap[ia, ib] = float(np.sum((la == ca) & (lb == cb)))
    matched = 0.0
    for _ in range(k):
        best = np.unravel_index(np.argmax(overlap), overlap.shape)
        matched += overlap[best]
        overlap[best[0], :] = -np.inf
        overlap[:, best[1]] = -np.inf
    denom = min(len(la), len(lb))
    return float(matched / denom) if denom > 0 else 0.0


def shuffled_alignment_null(labels_a: np.ndarray,
                             labels_b: np.ndarray,
                             n_shuffle: int = N_SHUFFLE_ALIGN,
                             seed: int      = SEED) -> dict:
    """
    Port of shuffled_alignment_score() from modularity.py / eval.py.
    Permute alive-neuron order in labels_a to build null distribution.
    """
    rng   = np.random.RandomState(seed)
    obs   = greedy_alignment_score(labels_a, labels_b)
    alive = (labels_a >= 0) & (labels_b >= 0)
    n     = alive.sum()

    nulls = np.zeros(n_shuffle)
    la    = labels_a.copy()
    for i in range(n_shuffle):
        la_perm              = labels_a.copy()
        alive_idx            = np.where(alive)[0]
        la_perm[alive_idx]   = labels_a[alive_idx[rng.permutation(n)]]
        nulls[i] = greedy_alignment_score(la_perm, labels_b)

    p_val = float((nulls >= obs).mean())
    z     = (obs - nulls.mean()) / (nulls.std() + 1e-12)
    return {'observed': obs, 'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std()), 'p_value': p_val, 'z_score': z}


def q_transfer(labels: np.ndarray, adj_other: np.ndarray) -> float:
    """
    Evaluate partition `labels` on a different adjacency matrix `adj_other`.
    Direct port of transfer_AaPb from eval.py (line 254):
      this_align_info['transfer_AaPb'] = girvan_newman(info_a['adj'], info_b['clusters'])
    """
    return gn_score(adj_other, labels)


def permutation_null_q(act_raw: np.ndarray,
                        observed_q: float,
                        null_graph: str = 'tune',   # 'tune' or 'down'
                        probe_weights_by_task: dict | None = None,
                        avail_tasks: list[str] | None = None,
                        n_perm: int  = N_PERM,
                        seed: int    = SEED) -> dict:
    """
    Permutation null for Q: shuffle act_raw columns → re-z-score → rebuild
    affinity → run the SAME spectral+MC optimizer used for observed Q.

    Uses reduced MC steps (MC_STEPS // 5) for speed.
    `null_graph` selects which association to use for the null:
      'tune' : task co-tuning graph (always available)
      'down' : probe-Jacobian graph (requires probe_weights_by_task)
    """
    rng     = np.random.RandomState(seed)
    null_qs = np.zeros(n_perm)

    for i in range(n_perm):
        act_perm = act_raw.copy()
        for col in range(act_perm.shape[1]):
            act_perm[:, col] = rng.permutation(act_perm[:, col])
        mu    = act_perm.mean(axis=1, keepdims=True)
        sigma = act_perm.std(axis=1,  keepdims=True) + 1e-8
        az    = (act_perm - mu) / sigma

        if null_graph == 'down' and probe_weights_by_task:
            A = downstream_probe_jacobian_affinity(probe_weights_by_task,
                                                    avail_tasks or [])
            if A is None:
                A = task_cotuning_affinity_matrix(az)
        else:
            A = task_cotuning_affinity_matrix(az)

        adj_sp = sparsify_affinity(A)
        # Same optimizer as observed Q (MC, not spectral-only)
        _, q_null, _ = cluster_modularity_from_adj(
            adj_sp,
            mc_steps=max(MC_STEPS // 5, 1000),
            seed=seed + i,
        )
        null_qs[i] = q_null

    p_val   = float((null_qs >= observed_q).mean())
    null_mu = float(null_qs.mean())
    null_sd = float(null_qs.std()) + 1e-12
    z       = (observed_q - null_mu) / null_sd
    return {'null_qs': null_qs, 'p_value': p_val,
            'z_score': z, 'null_mean': null_mu, 'null_std': null_sd}


def bootstrap_neuron_stability(act_z: np.ndarray,
                                full_labels: np.ndarray,
                                n_boot: int  = N_BOOT,
                                frac: float  = BOOT_FRAC,
                                seed: int    = SEED) -> dict:
    """
    Subsample `frac` of alive neurons, re-cluster, ARI(full, sub).
    Only compares alive neurons (labels >= 0).
    """
    rng   = np.random.RandomState(seed)
    alive = full_labels >= 0
    idxs  = np.where(alive)[0]
    n     = len(idxs)
    k     = int(n * frac)
    aris  = np.zeros(n_boot)

    for i in range(n_boot):
        sub_idx  = rng.choice(n, k, replace=False)
        sub_idxs = idxs[sub_idx]
        sub_z    = act_z[sub_idxs]
        adj_sub  = sparsify_affinity(task_cotuning_affinity_matrix(sub_z))
        lbl_s, _, _ = cluster_modularity_from_adj(
            adj_sub,
            mc_steps=max(MC_STEPS // 5, 500),
            seed=seed + i,
        )
        aris[i] = adjusted_rand_score(full_labels[sub_idxs], lbl_s)

    return {'aris': aris, 'mean_ari': float(aris.mean()),
            'std_ari': float(aris.std()),
            'p25_ari': float(np.percentile(aris, 25)),
            'p75_ari': float(np.percentile(aris, 75))}


def characterize_clusters(labels: np.ndarray,
                           act_z: np.ndarray,
                           avail_tasks: list[str],
                           si: np.ndarray | None) -> dict:
    """Auto-label each cluster. Ignores dead neurons (label == -1)."""
    stats = {}
    for cid in np.unique(labels[labels >= 0]):
        mask     = labels == cid
        mean_act = act_z[mask].mean(axis=0)

        m_idx = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
        a_idx = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]
        m_mean = float(mean_act[m_idx].mean()) if m_idx else 0.0
        a_mean = float(mean_act[a_idx].mean()) if a_idx else 0.0
        denom  = abs(m_mean) + abs(a_mean) + 1e-8
        sel    = (m_mean - a_mean) / denom
        label  = ('motion' if sel > GENERALIST_THRESH
                  else 'appearance' if sel < -GENERALIST_THRESH
                  else 'generalist')
        sharp  = float(mean_act.max() - mean_act.min())

        mean_si, std_si, si_dir = np.nan, np.nan, 'unknown'
        if si is not None and len(si) == D:
            si_c    = si[mask]
            mean_si = float(si_c.mean())
            std_si  = float(si_c.std())
            fd      = float((si_c > 0).mean())
            si_dir  = 'dorsal' if fd > 0.65 else 'ventral' if fd < 0.35 else 'mixed'

        stats[int(cid)] = {
            'label'       : label,
            'selectivity' : float(sel),
            'sharpness'   : sharp,
            'n_neurons'   : int(mask.sum()),
            'mean_act'    : mean_act.tolist(),
            'mean_si'     : mean_si,
            'std_si'      : std_si,
            'si_direction': si_dir,
        }
    return stats


def si_mwu_between_clusters(labels: np.ndarray, si: np.ndarray) -> dict:
    mwu = {}
    ids = np.unique(labels[labels >= 0])
    for i, c1 in enumerate(ids):
        for c2 in ids[i + 1:]:
            si1 = si[labels == c1]
            si2 = si[labels == c2]
            if len(si1) > 0 and len(si2) > 0:
                stat, p = mannwhitneyu(si1, si2, alternative='two-sided')
                mwu[(int(c1), int(c2))] = {'statistic': float(stat),
                                            'p_value'  : float(p)}
    return mwu


def motion_spearman(act_z, avail_tasks, si):
    m_idx = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
    a_idx = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]
    m = act_z[:, m_idx].mean(axis=1) if m_idx else np.zeros(D)
    a = act_z[:, a_idx].mean(axis=1) if a_idx else np.zeros(D)
    mot   = m - a
    valid = ~(np.isnan(mot) | np.isnan(si))
    if valid.sum() < 10:
        return np.nan, np.nan
    return spearmanr(mot[valid], si[valid])


# =============================================================================
# Main per-model runner
# =============================================================================

def run_model(model_name, cfg):
    print(f"\n{'='*65}")
    print(f"  {cfg['label']} — Downstream Modularity Clustering")
    print(f"  Associations: probe-Jacobian (down) + co-tuning (tune)")
    print(f"{'='*65}")

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(exist_ok=True)

    si_all          = load_si(cfg['si_path'])
    probe_w_paths   = cfg.get('probe_weights', {})
    has_probe_paths = bool(probe_w_paths)

    layer_ids    = []
    all_results  = {}

    for layer_idx in range(N_LAYERS):
        print(f"  L{layer_idx:02d}", end='', flush=True)

        act, act_z, avail_tasks = build_activation_matrix(cfg['tasks'], layer_idx)
        if act is None:
            print(" skip")
            continue

        si = si_all.get(layer_idx)

        # ── SECTION 1: ASSOCIATIONS ───────────────────────────────────────
        # Primary: probe-Jacobian downstream graph
        probe_weights_by_task = {}
        adj_down              = None
        if has_probe_paths:
            probe_weights_by_task = load_probe_weights(probe_w_paths, layer_idx,
                                                        avail_tasks)
            adj_down = downstream_probe_jacobian_affinity(probe_weights_by_task,
                                                           avail_tasks)

        # Secondary: task co-tuning graph
        adj_tune = task_cotuning_affinity_matrix(act_z)

        # ── SECTION 2: MODULARITY CLUSTERING ─────────────────────────────
        adj_tune_sp                          = sparsify_affinity(adj_tune)
        labels_tune, q_tune, k_tune, alive_t = cluster_modularity_alive(adj_tune_sp)

        labels_down, q_down, k_down, alive_d = None, np.nan, 0, None
        adj_down_sp                           = None
        if adj_down is not None:
            adj_down_sp                              = sparsify_affinity(adj_down)
            labels_down, q_down, k_down, alive_d    = cluster_modularity_alive(adj_down_sp)

        # ── SECTION 3: EVALUATION ─────────────────────────────────────────
        # 3a. Alignment and transfer between the two association graphs
        align_down_tune = None
        t_down_on_tune  = np.nan
        t_tune_on_down  = np.nan
        ari_down_tune   = np.nan

        if labels_down is not None:
            align_down_tune = shuffled_alignment_null(labels_down, labels_tune)
            t_down_on_tune  = q_transfer(labels_down, adj_tune_sp)
            t_tune_on_down  = q_transfer(labels_tune, adj_down_sp)
            valid           = (labels_down >= 0) & (labels_tune >= 0)
            if valid.sum() > 1:
                ari_down_tune = float(adjusted_rand_score(labels_down[valid],
                                                           labels_tune[valid]))

        # 3b. Permutation null for Q (same optimizer as observed)
        print(" [perm]", end='', flush=True)
        perm_tune = permutation_null_q(act, q_tune, null_graph='tune')
        perm_down = {'p_value': np.nan, 'z_score': np.nan,
                     'null_mean': np.nan, 'null_std': np.nan}
        if adj_down is not None:
            perm_down = permutation_null_q(
                act, q_down, null_graph='down',
                probe_weights_by_task=probe_weights_by_task,
                avail_tasks=avail_tasks,
            )

        # 3c. Bootstrap neuron stability (co-tuning graph)
        print("[boot]", end='', flush=True)
        boot = bootstrap_neuron_stability(act_z, labels_tune)

        # 3d. SI biological validation
        chars_tune  = characterize_clusters(labels_tune, act_z, avail_tasks, si)
        chars_down  = {}
        mwu_tune    = {}
        mwu_down    = {}
        rho_mot, p_mot = np.nan, np.nan

        if si is not None and len(si) == D:
            mwu_tune   = si_mwu_between_clusters(labels_tune, si)
            rho_mot, p_mot = motion_spearman(act_z, avail_tasks, si)
            if labels_down is not None:
                chars_down = characterize_clusters(labels_down, act_z, avail_tasks, si)
                mwu_down   = si_mwu_between_clusters(labels_down, si)

        layer_ids.append(layer_idx)
        all_results[layer_idx] = {
            # tune graph
            'q_tune'       : q_tune,
            'k_tune'       : k_tune,
            'perm_tune_p'  : perm_tune['p_value'],
            'perm_tune_z'  : perm_tune['z_score'],
            'n_dead_tune'  : int((~alive_t).sum()) if alive_t is not None else 0,
            # down graph
            'q_down'       : q_down,
            'k_down'       : k_down,
            'perm_down_p'  : perm_down['p_value'],
            'perm_down_z'  : perm_down['z_score'],
            'n_dead_down'  : int((~alive_d).sum()) if alive_d is not None else 0,
            # cross-graph evaluation
            'ari_down_tune'      : ari_down_tune,
            'align_obs'          : align_down_tune['observed'] if align_down_tune else np.nan,
            'align_p'            : align_down_tune['p_value']  if align_down_tune else np.nan,
            'align_z'            : align_down_tune['z_score']  if align_down_tune else np.nan,
            'transfer_down_tune' : t_down_on_tune,
            'transfer_tune_down' : t_tune_on_down,
            # bootstrap
            'boot_ari_mean': boot['mean_ari'],
            'boot_ari_std' : boot['std_ari'],
            # SI
            'rho_mot'      : float(rho_mot),
            'p_mot'        : float(p_mot),
            # cluster details
            'chars_tune'   : chars_tune,
            'chars_down'   : chars_down,
            'labels_tune'  : labels_tune,
            'labels_down'  : labels_down,
            'avail_tasks'  : avail_tasks,
        }

        # Console line
        down_str = (f" | down k={k_down} Q={q_down:.3f} "
                    f"perm_p={perm_down['p_value']:.2f}"
                    if labels_down is not None else " | down: no probe weights")
        print(f"\n    tune k={k_tune} Q={q_tune:.3f} "
              f"perm_p={perm_tune['p_value']:.2f}(z={perm_tune['z_score']:+.1f}) "
              f"boot={boot['mean_ari']:.2f}"
              f"{down_str} "
              f"rho_mot={rho_mot:+.3f}")
        if align_down_tune:
            print(f"    align(down,tune)={align_down_tune['observed']:.3f} "
                  f"p={align_down_tune['p_value']:.3f}  "
                  f"transfer(d→t)={t_down_on_tune:.3f} "
                  f"transfer(t→d)={t_tune_on_down:.3f}")

    if not layer_ids:
        print(f"  No usable layers for {model_name}")
        return

    _print_summary(model_name, cfg['label'], layer_ids, all_results, out_dir)
    _save_results(model_name, layer_ids, all_results, out_dir)


def _print_summary(model_name, label, layer_ids, all_results, out_dir):
    lines = [
        '=' * 100,
        f'  {label} — Downstream Modularity Clustering',
        f'  Affinity (down): Σ_tasks W_t.T @ W_t  (probe-Jacobian)',
        f'  Affinity (tune): max(0, act_z @ act_z.T)  (task co-tuning)',
        f'  Sparsify: top {GN_SPARSIFY_FRAC*100:.0f}% of edges (weighted)',
        f'  MC steps: {MC_STEPS}  |  Null: same optimizer, steps={max(MC_STEPS//5,1000)}',
        '=' * 100,
        '',
        '  SECTION A — ROBUSTNESS',
        f"  {'L':>3} {'k_t':>4} {'Q_t':>7} {'perm_p_t':>9} "
        f"{'k_d':>4} {'Q_d':>7} {'perm_p_d':>9} "
        f"{'boot':>7} {'ARI(d,t)':>9}",
        '  ' + '-' * 72,
    ]
    for li in layer_ids:
        r = all_results[li]
        lines.append(
            f"  {li:>3} {r['k_tune']:>4} {r['q_tune']:>7.4f} {r['perm_tune_p']:>9.3f} "
            f"{r['k_down']:>4} {r['q_down']:>7.4f} {r['perm_down_p']:>9.3f} "
            f"{r['boot_ari_mean']:>5.3f}±{r['boot_ari_std']:.2f} "
            f"{r['ari_down_tune']:>9.3f}"
        )
    lines += [
        '',
        '  perm_p < 0.05 → Q above random null (same MC optimizer)',
        '  boot > 0.6    → stable across 80% neuron subsamples',
        '  ARI(d,t)      → agreement between probe-Jacobian and co-tuning partitions',
        '',
        '  SECTION B — CROSS-GRAPH TRANSFER (eval.py transfer_AaPb analog)',
        f"  {'L':>3} {'align_obs':>10} {'align_p':>8} "
        f"{'Q(d→t)':>9} {'Q(t→d)':>9} {'rho_mot':>9}",
        '  ' + '-' * 55,
    ]
    for li in layer_ids:
        r = all_results[li]
        lines.append(
            f"  {li:>3} {r['align_obs']:>10.3f} {r['align_p']:>8.3f} "
            f"{r['transfer_down_tune']:>9.4f} {r['transfer_tune_down']:>9.4f} "
            f"{r['rho_mot']:>9.3f}"
        )
    lines += [
        '',
        '  Q(d→t) > 0 → probe-Jacobian partition captures structure in co-tuning graph',
        '  Q(t→d) > 0 → co-tuning partition captures structure in probe-Jacobian graph',
        '',
        '  SECTION C — CLUSTER LABELS (tune graph)',
        '',
    ]
    for li in layer_ids:
        r   = all_results[li]
        avt = r['avail_tasks']
        lines.append(f"  Layer {li:02d} [tune]:")
        for cid, cs in sorted(r['chars_tune'].items()):
            act_str = '  '.join(f"{t}:{v:+.2f}"
                                for t, v in zip(avt, cs['mean_act']))
            lines.append(
                f"    C{cid} [{cs['label']:>10}]  n={cs['n_neurons']:>4}  "
                f"sel={cs['selectivity']:+.3f}  sharp={cs['sharpness']:.3f}  "
                f"SI={cs['mean_si']:+.3f}  bio={cs['si_direction']}"
            )
            lines.append(f"         tuning: {act_str}")
        if r['chars_down']:
            lines.append(f"  Layer {li:02d} [down]:")
            for cid, cs in sorted(r['chars_down'].items()):
                act_str = '  '.join(f"{t}:{v:+.2f}"
                                    for t, v in zip(avt, cs['mean_act']))
                lines.append(
                    f"    C{cid} [{cs['label']:>10}]  n={cs['n_neurons']:>4}  "
                    f"sel={cs['selectivity']:+.3f}  sharp={cs['sharpness']:.3f}  "
                    f"SI={cs['mean_si']:+.3f}  bio={cs['si_direction']}"
                )
        lines.append('')

    lines += [
        '  Label guide:',
        '  motion     → fires more for MOTION_TASKS; expected SI direction: dorsal',
        '  appearance → fires more for APPEAR_TASKS; expected SI direction: ventral',
        '  generalist → flat z-scored profile, |selectivity| <= ' + str(GENERALIST_THRESH),
        '',
        f'  Reproduce:  python clustering_downstream.py {model_name}',
    ]
    txt = '\n'.join(lines)
    print(txt)
    with open(str(out_dir / f'{model_name}_downstream_summary.txt'), 'w') as f:
        f.write(txt)


def _save_results(model_name, layer_ids, all_results, out_dir):
    scalar_keys = [
        'q_tune', 'k_tune', 'perm_tune_p', 'perm_tune_z', 'n_dead_tune',
        'q_down', 'k_down', 'perm_down_p', 'perm_down_z', 'n_dead_down',
        'ari_down_tune', 'align_obs', 'align_p', 'align_z',
        'transfer_down_tune', 'transfer_tune_down',
        'boot_ari_mean', 'boot_ari_std', 'rho_mot', 'p_mot',
    ]
    save = dict(layer_ids=np.array(layer_ids))
    for k in scalar_keys:
        save[k] = np.array([all_results[l][k] for l in layer_ids], dtype=float)

    for li in layer_ids:
        if li in SELECTED_LAYERS:
            save[f'labels_tune_L{li:02d}']     = all_results[li]['labels_tune']
            save[f'chars_tune_json_L{li:02d}'] = np.array(
                [json.dumps(all_results[li]['chars_tune'])]
            )
            if all_results[li]['labels_down'] is not None:
                save[f'labels_down_L{li:02d}']     = all_results[li]['labels_down']
                save[f'chars_down_json_L{li:02d}'] = np.array(
                    [json.dumps(all_results[li]['chars_down'])]
                )

    path = out_dir / f'{model_name}_downstream_clustering.npz'
    np.savez_compressed(str(path), **save)
    print(f"\n  Saved: {path}")


# =============================================================================
# main
# =============================================================================

def main():
    args      = [a for a in sys.argv[1:] if not a.startswith('--')]
    model_arg = args[0] if args else None
    models    = [model_arg] if model_arg else list(MODEL_CONFIGS.keys())
    for model_name in models:
        run_model(model_name, MODEL_CONFIGS[model_name])
    print(f'\nAll done → {OUT_ROOT}/')


if __name__ == '__main__':
    main()
