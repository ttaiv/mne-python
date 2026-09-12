"""Call the public `mne.setup_source_space` with spacing=3 and measure the
actual nearest-neighbor vertex distances (mm) in the resulting SourceSpaces,
as an end-to-end sanity check on the hop-count `spacing` heuristic evaluated
in eval_spacing.py (which exercised the private decimation function
directly rather than the public API).

Usage:
    python eval_setup_source_space_spacing.py [subjects_dir]
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

import mne


def nearest_kept_distance(rr):
    """Distance (mm) from each vertex to its nearest other vertex."""
    tree = cKDTree(rr)
    d, _ = tree.query(rr, k=2)  # k=1 is the point itself, at distance 0
    return d[:, 1]


def report(name, rr_mm):
    nn = nearest_kept_distance(rr_mm)
    p = np.percentile(nn, [1, 5, 25, 50, 75, 95, 99])
    frac_below_half_median = np.mean(nn < 0.5 * np.median(nn))
    print(
        f"  {name}: n={len(rr_mm):5d} mean={nn.mean():.2f} std={nn.std():.2f} "
        f"min={nn.min():.2f} max={nn.max():.2f} "
        f"p1={p[0]:.2f} p5={p[1]:.2f} p25={p[2]:.2f} median={p[3]:.2f} "
        f"p75={p[4]:.2f} p95={p[5]:.2f} p99={p[6]:.2f} "
        f"frac<0.5*median={frac_below_half_median:.3f}"
    )


def main():
    subjects_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/tembe/mne_data/MNE-sample-data/subjects"
    )
    src = mne.setup_source_space(
        "sample", spacing=3, subjects_dir=subjects_dir, add_dist=False
    )

    print("=== setup_source_space(spacing=3) ===")
    all_rr_mm = []
    for hemi, s in zip(("lh", "rh"), src):
        rr_mm = s["rr"][s["vertno"]] * 1000.0  # stored in m, convert to mm
        report(hemi, rr_mm)
        all_rr_mm.append(rr_mm)
    report("both hemis combined", np.concatenate(all_rr_mm, axis=0))


if __name__ == "__main__":
    main()
