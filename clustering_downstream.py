"""
Downstream Modularity Clustering — with Robustness & Interpretation Validation
================================================================================

Answers two questions per layer per model:

  1. ARE THE CLUSTERS MEANINGFUL AND ROBUST?
     (a) Permutation null for Q
         Shuffle act_raw columns → re-z-score → re-cluster → null Q distribution.
         p_q = fraction of null runs with Q >= observed Q.
         Directly from modularity.py logic: shuffled_alignment_score() does
         the same shuffle for cluster alignment.

     (b) Bootstrap neuron stability
         Subsample 80% of neurons N_BOOT times → re-cluster → ARI(full, sub).
         High mean ARI = the partition is not driven by a handful of neurons.

     (c) Method agreement
         ARI(GN, Agglomerative) — two fundamentally different algorithms on
         the same data. Agreement = robust signal. Disagreement = one method
         is finding noise or topology the other misses.

     (d) Shuffled alignment null
         Port of shuffled_alignment_score() from modularity.py:
         permute neuron order in one clustering → compute alignment score →
         null distribution for how much agreement is expected by chance.

  2. WHAT DO THE CLUSTERS REFER TO?
     (a) Automatic labeling
         Each cluster gets a label from its z-scored tuning profile:
           'motion'     — mean z-score on MOTION_TASKS >> APPEAR_TASKS
           'appearance' — mean z-score on APPEAR_TASKS >> MOTION_TASKS
           'generalist' — flat profile (|selectivity| < GENERALIST_THRESH)
         Selectivity index per cluster:
           SI_task = (motion_mean - appear_mean) / (|motion_mean| + |appear_mean| + ε)
         Range [-1, 1]. Positive = motion-dominant, negative = appear-dominant.

     (b) SI alignment (biological validation — non-circular)
         For each cluster: mean SI (dorsal/ventral bias of its neurons).
         Mann-Whitney U test between all cluster pairs.
         Expected: motion-labeled cluster has SI > 0 (dorsal-biased).

     (c) Q transfer score (from modularity.py eval.py logic)
         Apply GN cluster partition to the AGGLOMERATIVE affinity graph and
         vice versa. If the partition is meaningful, it should score above
         random on an independently-constructed graph of the same neurons.
         transfer_GN_on_Agg = Q of GN labels evaluated on Agg adjacency matrix
         transfer_Agg_on_GN = Q of Agg labels evaluated on GN adjacency matrix

     (d) Tuning profile sharpness
         For each cluster: max(mean_act) - min(mean_act) in z-score units.
         Sharpness > 1 = the cluster has a clear task preference.
         Sharpness ~ 0 = flat generalist cluster.

Output per model:
    {model}_downstream_clustering.npz   — numeric results
    {model}_downstream_summary.txt      — human-readable table

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

AGG_LINKAGE = 'average'
DIST_THRESH = 0.4

GN_SPARSIFY_FRAC  = 0.20
GN_MAX_CLUSTERS   = 16
MC_STEPS          = 5000
GN_TARGET_ENTROPY = 0.15
GN_EPS            = 1e-15

# Robustness / interpretation parameters
N_PERM             = 100    # permutation null runs for Q
N_BOOT             = 100    # bootstrap stability runs
BOOT_FRAC          = 0.80   # fraction of neurons per bootstrap subsample
N_SHUFFLE_ALIGN    = 1000   # shuffled alignment null samples
GENERALIST_THRESH  = 0.20   # |selectivity| below this → 'generalist' label

MODEL_CONFIGS = {
    'vjepa_16f': {
        'label'  : 'V-JEPA2-16f',
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
# Affinity construction
# =============================================================================

def downstream_affinity_matrix(act_z: np.ndarray) -> np.ndarray:
    """A[i,j] = max(0, act_z[i] · act_z[j])  — rectified co-tuning Gram matrix."""
    gram = (act_z @ act_z.T).astype(np.float32)
    np.maximum(gram, 0.0, out=gram)
    np.fill_diagonal(gram, 0.0)
    return gram


def sparsify_affinity(adj: np.ndarray,
                      fraction: float = GN_SPARSIFY_FRAC) -> np.ndarray:
    adj = adj.copy()
    np.fill_diagonal(adj, 0.0)
    i_idx, j_idx = np.tril_indices(len(adj), k=-1)
    off_diag     = adj[i_idx, j_idx]
    nonzero      = off_diag[off_diag > 0]
    if len(nonzero) == 0:
        return np.zeros_like(adj)
    cutoff = float(np.quantile(nonzero, 1.0 - fraction))
    binary = np.where(adj >= cutoff, 1.0, 0.0).astype(np.float32)
    np.fill_diagonal(binary, 0.0)
    np.maximum(binary, binary.T, out=binary)
    return binary


def agg_affinity_matrix(act_z: np.ndarray) -> np.ndarray:
    """
    Convert agglomerative cosine distance to an affinity matrix
    A[i,j] = 1 - cosine_distance(act_z[i], act_z[j]).
    Used for Q transfer scoring (evaluate GN partition on Agg affinity).
    """
    from sklearn.preprocessing import normalize
    normed = normalize(act_z, norm='l2')
    sim    = (normed @ normed.T).astype(np.float32)
    np.maximum(sim, 0.0, out=sim)
    np.fill_diagonal(sim, 0.0)
    return sparsify_affinity(sim, fraction=GN_SPARSIFY_FRAC)


# =============================================================================
# GN modularity score
# =============================================================================

def gn_score(adj: np.ndarray, labels: np.ndarray) -> float:
    """Q = Σ_k [ e_kk - a_k² ]"""
    total = float(adj.sum())
    if total < GN_EPS:
        return 0.0
    Q = 0.0
    for c in np.unique(labels):
        mask = labels == c
        e_cc = float(adj[np.ix_(mask, mask)].sum()) / total
        a_c  = float(adj[mask].sum()) / total
        Q   += e_cc - a_c ** 2
    return Q


# =============================================================================
# Spectral initialisation
# =============================================================================

def spectral_modularity(adj: np.ndarray,
                         max_clusters: int = GN_MAX_CLUSTERS) -> np.ndarray:
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


# =============================================================================
# Monte Carlo refinement — vectorized delta-Q
# =============================================================================

def _entropy_to_temp(scores: np.ndarray, target: float,
                      init_t: float = 1.0, eps: float = 0.01,
                      max_steps: int = 500) -> float:
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


def monte_carlo_modularity(adj: np.ndarray,
                            labels_init: np.ndarray,
                            steps: int          = MC_STEPS,
                            target_entropy: float = GN_TARGET_ENTROPY,
                            seed: int            = SEED
                            ) -> tuple[np.ndarray, float]:
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
        mask      = labels == k
        e_kk      = float(adj[np.ix_(mask, mask)].sum()) / total
        a_k       = cluster_deg[k] / total
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
            best_q             = scores[best_ci]
            tmp                = labels.copy()
            tmp[i]             = candidates[best_ci]
            best_labels        = tmp

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
                e_kk      = float(row_sums[mk, k].sum()) / total
                a_k       = cluster_deg[k] / total
                current_q += e_kk - a_k ** 2

    unique = np.unique(best_labels)
    remap  = {int(old): new for new, old in enumerate(unique)}
    return np.array([remap[int(l)] for l in best_labels], dtype=int), float(best_q)


def cluster_downstream_modularity(act_z: np.ndarray,
                                   sparsify_frac: float  = GN_SPARSIFY_FRAC,
                                   mc_steps: int         = MC_STEPS,
                                   target_entropy: float = GN_TARGET_ENTROPY,
                                   seed: int             = SEED
                                   ) -> tuple[np.ndarray, float, int, np.ndarray]:
    """
    Returns: labels, gn_q, n_clusters, adj_sp (sparse adjacency kept for reuse)
    """
    adj    = downstream_affinity_matrix(act_z)
    adj_sp = sparsify_affinity(adj, fraction=sparsify_frac)
    if float(adj_sp.sum()) < GN_EPS:
        return np.zeros(len(act_z), dtype=int), 0.0, 1, adj_sp
    labels_spec  = spectral_modularity(adj_sp)
    labels_mc, q = monte_carlo_modularity(
        adj_sp, labels_spec,
        steps=mc_steps, target_entropy=target_entropy, seed=seed,
    )
    return labels_mc, q, int(labels_mc.max()) + 1, adj_sp


# =============================================================================
# Reference agglomerative clustering
# =============================================================================

def cluster_agglomerative_ref(act_z, dist_thresh=DIST_THRESH):
    condensed = np.clip(pdist(act_z, metric='cosine'), 0, 2)
    Z         = linkage(condensed, method=AGG_LINKAGE)
    return fcluster(Z, t=dist_thresh, criterion='distance') - 1


# =============================================================================
# ROBUSTNESS CHECK 1: Permutation null for Q
# — are the clusters better than random co-tuning structure?
# Mirrors the null logic from neuron_selectivity.py (threshold_stability_with_null)
# and the shuffled_alignment_score idea from modularity.py.
# =============================================================================

def permutation_null_q(act_raw: np.ndarray,
                        observed_q: float,
                        n_perm: int  = N_PERM,
                        seed: int    = SEED) -> dict:
    """
    Shuffle act_raw columns (tasks) → re-z-score → re-build affinity →
    spectral only (no MC — faster null) → record Q.

    Returns:
      null_qs   : (n_perm,) null Q values
      p_value   : fraction of null runs with Q >= observed_q
      z_score   : (observed_q - mean(null)) / std(null)
    """
    rng      = np.random.RandomState(seed)
    null_qs  = np.zeros(n_perm)

    for i in range(n_perm):
        act_perm = act_raw.copy()
        for col in range(act_perm.shape[1]):
            act_perm[:, col] = rng.permutation(act_perm[:, col])
        mu    = act_perm.mean(axis=1, keepdims=True)
        sigma = act_perm.std(axis=1,  keepdims=True) + 1e-8
        az    = (act_perm - mu) / sigma

        adj_sp       = sparsify_affinity(downstream_affinity_matrix(az))
        labels_spec  = spectral_modularity(adj_sp)
        null_qs[i]   = gn_score(adj_sp, labels_spec)

    p_val   = float((null_qs >= observed_q).mean())
    null_mu = float(null_qs.mean())
    null_sd = float(null_qs.std()) + 1e-12
    z       = (observed_q - null_mu) / null_sd

    return {'null_qs': null_qs, 'p_value': p_val, 'z_score': z,
            'null_mean': null_mu, 'null_std': null_sd}


# =============================================================================
# ROBUSTNESS CHECK 2: Bootstrap neuron stability
# — does the partition change when we subsample neurons?
# =============================================================================

def bootstrap_neuron_stability(act_z: np.ndarray,
                                full_labels: np.ndarray,
                                n_boot: int  = N_BOOT,
                                frac: float  = BOOT_FRAC,
                                seed: int    = SEED) -> dict:
    """
    Subsample `frac` of neurons N_BOOT times, re-cluster the subsample,
    then measure ARI between the subsampled-neuron assignments and the
    corresponding entries of full_labels.

    High mean ARI (> 0.6) → the partition is stable across neuron subsets.
    Low mean ARI           → the partition is driven by a small subset of
                             neurons or is otherwise fragile.
    """
    rng  = np.random.RandomState(seed)
    n    = len(act_z)
    k    = int(n * frac)
    aris = np.zeros(n_boot)

    for i in range(n_boot):
        idx      = rng.choice(n, k, replace=False)
        sub_z    = act_z[idx]
        adj_sub  = sparsify_affinity(downstream_affinity_matrix(sub_z))
        lbl_spec = spectral_modularity(adj_sub)
        lbl_mc, _ = monte_carlo_modularity(
            adj_sub, lbl_spec,
            steps=max(MC_STEPS // 5, 500),   # fewer steps for speed
            seed=seed + i,
        )
        aris[i] = adjusted_rand_score(full_labels[idx], lbl_mc)

    return {'aris': aris,
            'mean_ari': float(aris.mean()),
            'std_ari' : float(aris.std()),
            'p25_ari' : float(np.percentile(aris, 25)),
            'p75_ari' : float(np.percentile(aris, 75))}


# =============================================================================
# ROBUSTNESS CHECK 3: Shuffled alignment null
# Port of shuffled_alignment_score() from modularity.py.
# — is the agreement between GN and Agglomerative above chance?
# =============================================================================

def greedy_alignment_score(labels_a: np.ndarray,
                            labels_b: np.ndarray) -> float:
    """
    Greedy cluster matching: pair clusters across two partitions to maximize
    overlap, then compute fraction of neurons correctly matched.
    Port of alignment_score() from modularity.py, pure numpy.
    """
    ids_a = np.unique(labels_a)
    ids_b = np.unique(labels_b)
    k     = max(len(ids_a), len(ids_b))

    # Build overlap matrix
    overlap = np.zeros((k, k), dtype=np.float64)
    for ia, ca in enumerate(ids_a):
        for ib, cb in enumerate(ids_b):
            overlap[ia, ib] = float(np.sum((labels_a == ca) & (labels_b == cb)))

    # Greedy matching (same logic as greedy_alignment() in modularity.py)
    matched = 0.0
    for _ in range(k):
        best = np.unravel_index(np.argmax(overlap), overlap.shape)
        matched += overlap[best]
        overlap[best[0], :] = -np.inf
        overlap[:, best[1]] = -np.inf

    denom = min(len(labels_a), len(labels_b))
    return float(matched / denom) if denom > 0 else 0.0


def shuffled_alignment_null(labels_a: np.ndarray,
                             labels_b: np.ndarray,
                             n_shuffle: int = N_SHUFFLE_ALIGN,
                             seed: int      = SEED) -> dict:
    """
    Port of shuffled_alignment_score() from modularity.py.
    Permute labels_a neuron order n_shuffle times, compute alignment score
    against labels_b each time → null distribution.

    Observed alignment significantly above null → the two methods agree
    more than expected by chance → partition is robust.
    """
    rng     = np.random.RandomState(seed)
    obs     = greedy_alignment_score(labels_a, labels_b)
    n       = len(labels_a)
    nulls   = np.array([
        greedy_alignment_score(labels_a[rng.permutation(n)], labels_b)
        for _ in range(n_shuffle)
    ])
    p_val   = float((nulls >= obs).mean())
    z       = (obs - nulls.mean()) / (nulls.std() + 1e-12)
    return {'observed': obs, 'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std()), 'p_value': p_val, 'z_score': z}


# =============================================================================
# INTERPRETATION: Q transfer score
# From eval.py in the modularity repo ("transfer_AaPb" metric):
# evaluate partition A on adjacency matrix B.
# If the partition is truly meaningful it should score > 0 on an
# independently-constructed affinity graph of the same neurons.
# =============================================================================

def q_transfer(labels: np.ndarray, adj_other: np.ndarray) -> float:
    """
    Compute GN Q score of `labels` evaluated on `adj_other`.
    High score → the partition is meaningful on a different affinity graph,
    i.e., it captures structure that generalises beyond one construction method.
    """
    return gn_score(adj_other, labels)


# =============================================================================
# INTERPRETATION: Cluster labeling and selectivity
# =============================================================================

def label_cluster(mean_act: np.ndarray,
                  avail_tasks: list[str]) -> tuple[str, float]:
    """
    Assign a semantic label and selectivity index to a cluster.

    Selectivity index:
      sel = (motion_mean - appear_mean) / (|motion_mean| + |appear_mean| + ε)
      Range [-1, 1].
      sel > +GENERALIST_THRESH → 'motion'
      sel < -GENERALIST_THRESH → 'appearance'
      |sel| <= GENERALIST_THRESH → 'generalist'

    'motion' here follows MOTION_TASKS / APPEAR_TASKS defined at module level.
    """
    m_idx = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
    a_idx = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]

    m_mean = float(mean_act[m_idx].mean()) if m_idx else 0.0
    a_mean = float(mean_act[a_idx].mean()) if a_idx else 0.0

    denom = abs(m_mean) + abs(a_mean) + 1e-8
    sel   = (m_mean - a_mean) / denom

    if sel > GENERALIST_THRESH:
        label = 'motion'
    elif sel < -GENERALIST_THRESH:
        label = 'appearance'
    else:
        label = 'generalist'

    return label, float(sel)


def tuning_sharpness(mean_act: np.ndarray) -> float:
    """max - min of z-scored tuning profile. > 1 = clear task preference."""
    return float(mean_act.max() - mean_act.min())


def characterize_clusters(labels: np.ndarray,
                           act_z: np.ndarray,
                           avail_tasks: list[str],
                           si: np.ndarray | None) -> dict:
    """
    For each cluster produce:
      label        : 'motion' | 'appearance' | 'generalist'
      selectivity  : float in [-1, 1]  (positive = motion-dominant)
      sharpness    : float  (max - min of mean z-scored tuning profile)
      n_neurons    : int
      mean_act     : (n_tasks,) mean z-scored activation per task
      mean_si      : float (mean biological SI of member neurons)
      std_si       : float
      si_direction : 'dorsal' | 'ventral' | 'mixed' | 'unknown'
    """
    stats = {}
    for cid in np.unique(labels):
        mask     = labels == cid
        mean_act = act_z[mask].mean(axis=0)
        lbl, sel = label_cluster(mean_act, avail_tasks)
        sharp    = tuning_sharpness(mean_act)

        mean_si, std_si, si_dir = np.nan, np.nan, 'unknown'
        if si is not None and len(si) == D:
            si_c    = si[mask]
            mean_si = float(si_c.mean())
            std_si  = float(si_c.std())
            # SI > 0 = dorsal-biased, SI < 0 = ventral-biased
            frac_dorsal = float((si_c > 0).mean())
            if frac_dorsal > 0.65:
                si_dir = 'dorsal'
            elif frac_dorsal < 0.35:
                si_dir = 'ventral'
            else:
                si_dir = 'mixed'

        stats[int(cid)] = {
            'label'       : lbl,
            'selectivity' : sel,
            'sharpness'   : sharp,
            'n_neurons'   : int(mask.sum()),
            'mean_act'    : mean_act.tolist(),
            'mean_si'     : mean_si,
            'std_si'      : std_si,
            'si_direction': si_dir,
        }
    return stats


def si_mwu_between_clusters(labels: np.ndarray,
                             si: np.ndarray) -> dict:
    """Mann-Whitney U between all pairs of clusters on their SI distributions."""
    mwu = {}
    ids = np.unique(labels)
    for i, c1 in enumerate(ids):
        for c2 in ids[i + 1:]:
            si1, si2 = si[labels == c1], si[labels == c2]
            if len(si1) > 0 and len(si2) > 0:
                stat, p = mannwhitneyu(si1, si2, alternative='two-sided')
                mwu[(int(c1), int(c2))] = {'statistic': float(stat),
                                            'p_value'  : float(p)}
    return mwu


def motion_spearman(act_z, avail_tasks, si):
    """Spearman rho(per-neuron motion score, SI)."""
    m_idx = [i for i, t in enumerate(avail_tasks) if t in MOTION_TASKS]
    a_idx = [i for i, t in enumerate(avail_tasks) if t in APPEAR_TASKS]
    m = act_z[:, m_idx].mean(axis=1) if m_idx else np.zeros(D)
    a = act_z[:, a_idx].mean(axis=1) if a_idx else np.zeros(D)
    mot = m - a
    valid = ~(np.isnan(mot) | np.isnan(si))
    if valid.sum() < 10:
        return np.nan, np.nan
    return spearmanr(mot[valid], si[valid])


# =============================================================================
# Main per-model runner
# =============================================================================

def run_model(model_name, cfg):
    print(f"\n{'='*60}")
    print(f"  {cfg['label']} — Downstream Modularity Clustering")
    print(f"  + Robustness & Interpretation Validation")
    print(f"{'='*60}")

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(exist_ok=True)

    si_all = load_si(cfg['si_path'])

    # Accumulators across layers
    layer_ids      = []
    all_results    = {}   # keyed by layer_idx, stores everything

    for layer_idx in range(N_LAYERS):
        print(f"  L{layer_idx:02d}", end='', flush=True)

        act, act_z, avail_tasks = build_activation_matrix(cfg['tasks'], layer_idx)
        if act is None:
            print(" skip")
            continue

        si = si_all.get(layer_idx)

        # ── Downstream GN clustering ──────────────────────────────────────
        labels_gn, q_gn, n_gn, adj_gn = cluster_downstream_modularity(act_z)

        # ── Reference agglomerative ───────────────────────────────────────
        labels_agg   = cluster_agglomerative_ref(act_z)
        adj_agg      = agg_affinity_matrix(act_z)
        ari_agg_gn   = float(adjusted_rand_score(labels_agg, labels_gn))

        # ── ROBUSTNESS 1: Permutation null for Q ─────────────────────────
        print(" [perm]", end='', flush=True)
        perm = permutation_null_q(act, q_gn)

        # ── ROBUSTNESS 2: Bootstrap neuron stability ──────────────────────
        print("[boot]", end='', flush=True)
        boot = bootstrap_neuron_stability(act_z, labels_gn)

        # ── ROBUSTNESS 3: Shuffled alignment null ─────────────────────────
        align_null = shuffled_alignment_null(labels_gn, labels_agg)

        # ── INTERPRETATION: Q transfer ────────────────────────────────────
        # GN partition evaluated on Agg affinity and vice versa
        q_gn_on_agg  = q_transfer(labels_gn,  adj_agg)
        q_agg_on_gn  = q_transfer(labels_agg, adj_gn)

        # ── INTERPRETATION: Cluster labeling ─────────────────────────────
        cluster_chars = characterize_clusters(labels_gn, act_z, avail_tasks, si)

        # ── INTERPRETATION: SI Mann-Whitney between clusters ──────────────
        mwu = {}
        if si is not None and len(si) == D:
            mwu = si_mwu_between_clusters(labels_gn, si)

        # ── INTERPRETATION: Spearman rho(motion, SI) ─────────────────────
        rho_mot, p_mot = (np.nan, np.nan)
        if si is not None and len(si) == D:
            rho_mot, p_mot = motion_spearman(act_z, avail_tasks, si)

        layer_ids.append(layer_idx)
        all_results[layer_idx] = {
            'q_gn'         : q_gn,
            'n_gn'         : n_gn,
            'n_agg'        : int(labels_agg.max()) + 1,
            'ari_agg_gn'   : ari_agg_gn,
            # robustness
            'perm_p'       : perm['p_value'],
            'perm_z'       : perm['z_score'],
            'boot_ari_mean': boot['mean_ari'],
            'boot_ari_std' : boot['std_ari'],
            'align_obs'    : align_null['observed'],
            'align_p'      : align_null['p_value'],
            'align_z'      : align_null['z_score'],
            # interpretation
            'q_gn_on_agg'  : q_gn_on_agg,
            'q_agg_on_gn'  : q_agg_on_gn,
            'cluster_chars': cluster_chars,
            'mwu'          : mwu,
            'rho_mot'      : float(rho_mot),
            'p_mot'        : float(p_mot),
            'labels_gn'    : labels_gn,
            'avail_tasks'  : avail_tasks,
        }

        # Console line
        char_str = '  '.join(
            f"C{c}:{s['label']}(sel={s['selectivity']:+.2f}"
            f" sharp={s['sharpness']:.2f}"
            f" si={s['mean_si']:+.2f}[{s['si_direction']}])"
            for c, s in cluster_chars.items()
        )
        print(f"\n    k={n_gn} Q={q_gn:.3f} "
              f"perm_p={perm['p_value']:.3f}(z={perm['z_score']:+.1f}) "
              f"boot_ARI={boot['mean_ari']:.3f}±{boot['std_ari']:.3f} "
              f"align_p={align_null['p_value']:.3f} "
              f"ARI(GN,Agg)={ari_agg_gn:.3f} "
              f"rho_mot={rho_mot:+.3f}")
        print(f"    transfer: Q(GN→Agg)={q_gn_on_agg:.3f}  "
              f"Q(Agg→GN)={q_agg_on_gn:.3f}")
        print(f"    {char_str}")

    if not layer_ids:
        print(f"  No usable layers for {model_name}")
        return

    _print_summary(model_name, cfg['label'], layer_ids, all_results, out_dir)
    _save_results(model_name, layer_ids, all_results, out_dir)


def _print_summary(model_name, label, layer_ids, all_results, out_dir):
    """
    Three-section table:
      Section A — robustness: perm_p, boot_ARI, align_p, ARI(GN, Agg)
      Section B — interpretation: Q transfer, rho_mot
      Section C — per-cluster labels, selectivity, sharpness, SI direction
    """
    lines = [
        '=' * 100,
        f'  {label} — Downstream Modularity Clustering',
        f'  Affinity: A[i,j] = max(0, act_z[i]·act_z[j])  |  '
        f'Sparsify: top {GN_SPARSIFY_FRAC*100:.0f}%  |  MC steps: {MC_STEPS}',
        '=' * 100,
        '',
        '  SECTION A — ROBUSTNESS',
        '  (Are the clusters meaningful structure rather than noise?)',
        '',
        f"  {'L':>3} {'k':>3} {'Q':>7} "
        f"{'perm_p':>7} {'perm_z':>7} "
        f"{'boot_ARI':>9} "
        f"{'align_p':>8} {'align_z':>7} "
        f"{'ARI(GN,Agg)':>12}",
        '  ' + '-' * 70,
    ]
    for li in layer_ids:
        r = all_results[li]
        lines.append(
            f"  {li:>3} {r['n_gn']:>3} {r['q_gn']:>7.4f} "
            f"{r['perm_p']:>7.3f} {r['perm_z']:>7.2f} "
            f"{r['boot_ari_mean']:>7.3f}±{r['boot_ari_std']:.2f} "
            f"{r['align_p']:>8.3f} {r['align_z']:>7.2f} "
            f"{r['ari_agg_gn']:>12.3f}"
        )
    lines += [
        '',
        '  Robustness guide:',
        '  perm_p < 0.05  → Q is above the random co-tuning null',
        '  perm_z > 2     → Q is > 2 SD above null mean',
        '  boot_ARI > 0.6 → partition stable across 80% neuron subsamples',
        '  align_p < 0.05 → GN/Agg agreement exceeds chance (shuffled null)',
        '  ARI(GN,Agg) > 0.5 → both methods find the same partition',
        '',
        '  SECTION B — INTERPRETATION (Q TRANSFER + SI CORRELATION)',
        '  (Does the partition generalise, and does it align with biology?)',
        '',
        f"  {'L':>3} {'Q(GN→Agg)':>11} {'Q(Agg→GN)':>11} {'rho_mot':>9} {'p_mot':>9}",
        '  ' + '-' * 46,
    ]
    for li in layer_ids:
        r = all_results[li]
        lines.append(
            f"  {li:>3} {r['q_gn_on_agg']:>11.4f} {r['q_agg_on_gn']:>11.4f} "
            f"{r['rho_mot']:>9.3f} {r['p_mot']:>9.4f}"
        )
    lines += [
        '',
        '  Transfer guide:',
        '  Q(GN→Agg) > 0  → GN partition captures real structure in the Agg graph',
        '  Q(Agg→GN) > 0  → Agg partition captures real structure in the GN graph',
        '  High transfer in both directions → robust cross-method structure',
        '  rho_mot > 0    → motion-active neurons are dorsal-biased (expected)',
        '',
        '  SECTION C — CLUSTER LABELS & SI ALIGNMENT',
        '  (What does each cluster represent?)',
        '',
    ]
    for li in layer_ids:
        r    = all_results[li]
        avt  = r['avail_tasks']
        lines.append(f"  Layer {li:02d}:")
        for cid, cs in sorted(r['cluster_chars'].items()):
            act_str = '  '.join(
                f"{t}:{v:+.2f}"
                for t, v in zip(avt, cs['mean_act'])
            )
            lines.append(
                f"    C{cid} [{cs['label']:>10}]  "
                f"n={cs['n_neurons']:>4}  "
                f"sel={cs['selectivity']:+.3f}  "
                f"sharp={cs['sharpness']:.3f}  "
                f"SI={cs['mean_si']:+.3f}±{cs['std_si']:.3f}  "
                f"bio={cs['si_direction']}"
            )
            lines.append(f"         tuning: {act_str}")
        lines.append('')
    lines += [
        '  Cluster label guide:',
        '  motion     — fires more for MOTION_TASKS than APPEAR_TASKS',
        '               consistent with dorsal (SI > 0) if biology aligns',
        '  appearance — fires more for APPEAR_TASKS than MOTION_TASKS',
        '               consistent with ventral (SI < 0) if biology aligns',
        '  generalist — flat z-scored profile; |selectivity| <= '
        f'{GENERALIST_THRESH}',
        '  sharpness > 1.0  → strong task preference',
        '  sharpness < 0.5  → weak/flat tuning (may be noise cluster)',
        '',
        f'  Reproduce:  python clustering_downstream.py {model_name}',
    ]
    txt = '\n'.join(lines)
    print(txt)
    with open(str(out_dir / f'{model_name}_downstream_summary.txt'), 'w') as f:
        f.write(txt)


def _save_results(model_name, layer_ids, all_results, out_dir):
    save = dict(layer_ids=np.array(layer_ids))
    scalar_keys = [
        'q_gn', 'n_gn', 'n_agg', 'ari_agg_gn',
        'perm_p', 'perm_z', 'boot_ari_mean', 'boot_ari_std',
        'align_obs', 'align_p', 'align_z',
        'q_gn_on_agg', 'q_agg_on_gn', 'rho_mot', 'p_mot',
    ]
    for k in scalar_keys:
        save[k] = np.array([all_results[l][k] for l in layer_ids], dtype=float)

    for li in layer_ids:
        if li in SELECTED_LAYERS:
            save[f'labels_gn_L{li:02d}']    = all_results[li]['labels_gn']
            save[f'cluster_json_L{li:02d}'] = np.array(
                [json.dumps(all_results[li]['cluster_chars'])]
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
