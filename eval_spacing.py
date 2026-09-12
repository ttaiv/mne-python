"""Evaluate how well the hop-count `spacing` heuristic (used by
`setup_source_space` and currently reused by `setup_subcortical_source_space`)
approximates a real mm spacing, on a subcortical (marching-cubes-like) mesh
vs. a regular FreeSurfer cortical mesh.

Requires an editable mne install with scipy available, e.g.:
    uv venv .venv-test && uv pip install --python .venv-test -e . scipy

Usage:
    python eval_spacing.py [subjects_dir]
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

from mne.surface import (
    _decimate_surface_spacing,
    _keep_largest_component,
    complete_surface_info,
    read_surface,
)

SPACINGS = (2, 3, 5, 8, 10)


def edge_lengths(rr, tris):
    """Euclidean length (mm) of every mesh edge, before any decimation."""
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    return np.linalg.norm(rr[edges[:, 0]] - rr[edges[:, 1]], axis=1)


def nearest_kept_distance(rr):
    """Distance (mm) from each kept vertex to its nearest other kept vertex."""
    tree = cKDTree(rr)
    d, _ = tree.query(rr, k=2)  # k=1 is the point itself, at distance 0
    return d[:, 1]


def evaluate(name, fname, keep_largest_component=True):
    rr, tris = read_surface(fname)[:2]
    rr = np.asarray(rr, float)
    tris = np.asarray(tris, np.int64)
    if keep_largest_component:
        rr, tris = _keep_largest_component(rr, tris)

    surf = dict(rr=rr, tris=tris)
    complete_surface_info(surf, do_neighbor_vert=True, copy=False)

    el = edge_lengths(rr, tris)
    print(f"=== {name} ===")
    print(f"n_vertices={len(rr)} n_tris={len(tris)}")
    print(
        f"edge length (mm): mean={el.mean():.3f} std={el.std():.3f} "
        f"min={el.min():.3f} max={el.max():.3f} cv={el.std() / el.mean():.3f}"
    )

    for spacing in SPACINGS:
        s = dict(surf)  # _decimate_surface_spacing mutates "inuse" in place
        _decimate_surface_spacing(s, spacing)
        vertno = np.where(s["inuse"])[0]
        nn = nearest_kept_distance(rr[vertno])
        print(
            f"  spacing={spacing:2d} -> n_kept={len(vertno):5d} "
            f"actual NN dist (mm): mean={nn.mean():.2f} std={nn.std():.2f} "
            f"min={nn.min():.2f} max={nn.max():.2f} "
            f"(max/min ratio={nn.max() / nn.min():.1f})"
        )
    print()


def main():
    subjects_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/tembe/mne_data/MNE-sample-data/subjects"
    )
    evaluate(
        "cerebellum_sparse.white (subcortical, marching-cubes-like)",
        f"{subjects_dir}/sample/surf/cerebellum_sparse.white",
        keep_largest_component=True,
    )
    evaluate(
        "lh.white (cortical, FreeSurfer regular tessellation)",
        f"{subjects_dir}/sample/surf/lh.white",
        keep_largest_component=False,
    )


if __name__ == "__main__":
    main()
