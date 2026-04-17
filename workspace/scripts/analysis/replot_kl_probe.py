#!/usr/bin/env python
"""Standalone re-plot tool for KL-by-position diagnostic output.

Reads a session's ``kl_plots/*.npz`` files and regenerates the PDFs via the
shared plotting functions in ``kl_probe.py``. Useful for:
  - Re-styling plots without rerunning training.
  - Regenerating the overlay plot after changing stratum colors.
  - Lowering ``min_tokens_per_bin`` to unsuppress late-position bins.

Usage:
    python workspace/scripts/analysis/replot_kl_probe.py \\
        outputs/2026-04-17/10-20-30/kl_plots/

    # Lower sparse-bin threshold:
    python workspace/scripts/analysis/replot_kl_probe.py \\
        outputs/.../kl_plots/ --min-tokens-per-bin 10
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def _setup_import_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    src_path = os.path.join(repo_root, "workspace", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_STEP_RE = re.compile(r"step(\d+)_(?!raw_trajectories)(.*?)\.npz$")


def main() -> int:
    _setup_import_path()
    from self_distill_hybrid.kl_probe import plot_from_npz, overlay_from_npz_dir, STRATA

    parser = argparse.ArgumentParser(description="Regenerate KL probe PDFs from .npz")
    parser.add_argument(
        "plots_dir",
        help="Directory containing step*_*.npz files (usually outputs/<session>/kl_plots/).",
    )
    parser.add_argument(
        "--min-tokens-per-bin",
        type=int,
        default=30,
        help="Threshold for sparse-bin suppression when rendering (default: 30).",
    )
    args = parser.parse_args()

    plots_dir = os.path.abspath(args.plots_dir)
    if not os.path.isdir(plots_dir):
        print(f"error: {plots_dir} is not a directory", file=sys.stderr)
        return 2

    # Single-stratum PDFs from each step*_{stratum}.npz.
    n_single = 0
    steps_seen: set[int] = set()
    for npz_path in sorted(glob.glob(os.path.join(plots_dir, "step*.npz"))):
        base = os.path.basename(npz_path)
        m = _STEP_RE.match(base)
        if not m:
            continue
        step = int(m.group(1))
        stratum = m.group(2)
        if stratum not in STRATA:
            continue
        pdf_path = os.path.join(plots_dir, base.replace(".npz", ".pdf"))
        try:
            plot_from_npz(npz_path, pdf_path, min_tokens_per_bin=args.min_tokens_per_bin)
            n_single += 1
            steps_seen.add(step)
        except Exception as e:
            print(f"  [skip] {base}: {e}", file=sys.stderr)

    # Overlay PDFs — one per step, reconstructed from the 4 stratum .npz files.
    n_overlay = 0
    for step in sorted(steps_seen):
        overlay_pdf = os.path.join(plots_dir, f"step{step:03d}_overlay.pdf")
        try:
            overlay_from_npz_dir(
                plots_dir, step, overlay_pdf,
                min_tokens_per_bin=args.min_tokens_per_bin,
            )
            n_overlay += 1
        except Exception as e:
            print(f"  [skip overlay step {step}]: {e}", file=sys.stderr)

    print(f"rendered {n_single} per-stratum PDFs and {n_overlay} overlay PDFs in {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
