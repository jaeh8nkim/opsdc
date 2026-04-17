"""KL-by-position probe: training-time diagnostic for OPSD.

Accumulates per-token reverse-KL values during training, stratifies by
correctness and truncation status, bins by absolute token position, and
renders PDF plots at flush time.

Output per flush:
  - 4 per-stratum PDFs + 4 .npz (ended_correct / ended_incorrect / truncated / all)
  - 1 overlay PDF (the 4 stratum curves layered in a single axes)
  - 1 consolidated raw_trajectories.npz (per-rollout KL arrays, for ad-hoc
    offline analysis)

Also hosts ``snap_to_boundary`` and ``find_think_close`` since they're shared
with the reinjection pipeline.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KLProbeConfig:
    enabled: bool = True
    plot_freq: int = 50
    n_abs_bins: int = 64
    max_position: int = 8192
    min_tokens_per_bin: int = 30
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    raw_exemplar_count: int = 32


STRATA = ("ended_correct", "ended_incorrect", "truncated", "all")
STRATUM_COLORS = {
    "ended_correct": "#2b9e3f",
    "ended_incorrect": "#c0392b",
    "truncated": "#7f7f7f",
    "all": "#1f77b4",
}


# ---------------------------------------------------------------------------
# Per-rollout record
# ---------------------------------------------------------------------------


@dataclass
class RolloutRecord:
    kl: np.ndarray  # (L,) float32 — per-token reverse KL
    length: int
    correct: bool
    truncated: bool

    @property
    def stratum(self) -> str:
        if self.truncated:
            return "truncated"
        return "ended_correct" if self.correct else "ended_incorrect"


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class KLProbeAccumulator:
    """Accumulates per-rollout KL trajectories across training steps.

    Usage:
        acc = KLProbeAccumulator(cfg)
        # per step:
        acc.add_step(kl_per_token, student_lengths, correct_mask, truncated_mask)
        # at flush:
        acc.flush(output_dir="outputs/<session>/kl_plots", step=50, meta={...})
        acc.reset()
    """

    def __init__(self, cfg: KLProbeConfig):
        self.cfg = cfg
        self.rollouts: list[RolloutRecord] = []
        # Track the most recent batch separately for raw-exemplar plots at flush time.
        self._latest_batch: list[RolloutRecord] = []

    def has_data(self) -> bool:
        return bool(self.rollouts)

    def reset(self) -> None:
        self.rollouts.clear()
        self._latest_batch.clear()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_step(
        self,
        kl_per_token: torch.Tensor,
        student_lengths: list[int],
        correct_mask: list[bool],
        truncated_mask: list[bool],
    ) -> None:
        """Ingest one training step's per-token KL values.

        Args:
            kl_per_token: (N_total_tokens,) flat float tensor — KL for every
                student response token across the batch, concatenated in
                sample-major order.
            student_lengths: list of int, one per sample — length of the student's
                response (same ordering as the concatenation).
            correct_mask: list of bool, one per sample.
            truncated_mask: list of bool, one per sample.
        """
        if not self.cfg.enabled:
            return

        kl_np = kl_per_token.detach().float().cpu().numpy()
        expected = sum(student_lengths)
        if kl_np.shape[0] != expected:
            logger.warning(
                "KL probe: kl_per_token length %d != sum(student_lengths) %d — skipping step",
                kl_np.shape[0],
                expected,
            )
            return

        batch_records: list[RolloutRecord] = []
        cursor = 0
        for L, correct, trunc in zip(student_lengths, correct_mask, truncated_mask):
            if L <= 0:
                cursor += max(L, 0)
                continue
            chunk = kl_np[cursor : cursor + L].copy()
            cursor += L
            rec = RolloutRecord(
                kl=chunk,
                length=int(L),
                correct=bool(correct),
                truncated=bool(trunc),
            )
            self.rollouts.append(rec)
            batch_records.append(rec)

        # Remember the latest batch for raw-exemplar plots.
        self._latest_batch = batch_records

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _stratum_rollouts(self, stratum: str) -> list[RolloutRecord]:
        if stratum == "all":
            return list(self.rollouts)
        return [r for r in self.rollouts if r.stratum == stratum]

    def _compute_bins(self, rollouts: list[RolloutRecord]) -> dict:
        """Return bin_edges, bin_means, bin_counts, survival_counts, bin_ci_low, bin_ci_high.

        Absolute-position binning only (64 bins × 128 tokens over [0, 8192]).
        """
        cfg = self.cfg
        n_bins = cfg.n_abs_bins
        edges = np.linspace(0.0, float(cfg.max_position), n_bins + 1)

        # Per-rollout → (bin_idx array, kl array). We retain the per-rollout
        # structure for bootstrap resampling over rollouts (not tokens).
        per_rollout_bins: list[np.ndarray] = []
        per_rollout_kl: list[np.ndarray] = []
        for r in rollouts:
            if r.length <= 0:
                per_rollout_bins.append(np.zeros(0, dtype=np.int64))
                per_rollout_kl.append(np.zeros(0, dtype=np.float32))
                continue
            positions = np.arange(r.length, dtype=np.float32)
            # np.digitize returns bin indices in [1, n_bins+1]; subtract 1 → [0, n_bins].
            idx = np.clip(np.digitize(positions, edges[1:-1]), 0, n_bins - 1)
            per_rollout_bins.append(idx.astype(np.int64))
            per_rollout_kl.append(r.kl.astype(np.float32))

        if per_rollout_bins:
            all_bins = np.concatenate(per_rollout_bins)
            all_kl = np.concatenate(per_rollout_kl)
        else:
            all_bins = np.zeros(0, dtype=np.int64)
            all_kl = np.zeros(0, dtype=np.float32)

        bin_sums = np.bincount(all_bins, weights=all_kl, minlength=n_bins)
        bin_counts = np.bincount(all_bins, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            bin_means = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)

        # Survival: how many rollouts reach each bin boundary. Flags the
        # late-position bins where conclusions rest on a shrinking set of long
        # rollouts (selection bias).
        lengths = np.array([r.length for r in rollouts], dtype=np.int64)
        survival = np.zeros(n_bins, dtype=np.int64)
        for i in range(n_bins):
            survival[i] = int(np.sum(lengths > edges[i]))

        # Bootstrap CI — clustered over rollouts.
        if cfg.bootstrap_resamples > 0 and len(rollouts) >= 2:
            ci_low, ci_high = self._bootstrap_ci(
                per_rollout_bins, per_rollout_kl, n_bins, cfg.bootstrap_resamples
            )
        else:
            ci_low = np.full(n_bins, np.nan)
            ci_high = np.full(n_bins, np.nan)

        return {
            "bin_edges": edges,
            "bin_means": bin_means,
            "bin_counts": bin_counts,
            "survival": survival,
            "bin_ci_low": ci_low,
            "bin_ci_high": ci_high,
            "n_rollouts": len(rollouts),
        }

    def _bootstrap_ci(
        self,
        per_rollout_bins: list[np.ndarray],
        per_rollout_kl: list[np.ndarray],
        n_bins: int,
        n_resamples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """95% clustered bootstrap CI over rollouts."""
        rng = np.random.default_rng(self.cfg.bootstrap_seed)
        R = len(per_rollout_bins)
        means = np.full((n_resamples, n_bins), np.nan, dtype=np.float32)
        for b in range(n_resamples):
            sample_idx = rng.integers(0, R, size=R)
            sums = np.zeros(n_bins, dtype=np.float64)
            counts = np.zeros(n_bins, dtype=np.int64)
            for r_idx in sample_idx:
                bins_r = per_rollout_bins[r_idx]
                kl_r = per_rollout_kl[r_idx]
                if bins_r.size == 0:
                    continue
                sums += np.bincount(bins_r, weights=kl_r, minlength=n_bins)
                counts += np.bincount(bins_r, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                means[b] = np.where(counts > 0, sums / counts, np.nan)
        # Percentiles, ignoring NaNs. Suppress expected "All-NaN slice" runtime
        # warnings that numpy emits for bins with no data — they're informational
        # and handled by min_tokens_per_bin suppression downstream.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
            ci_low = np.nanpercentile(means, 2.5, axis=0)
            ci_high = np.nanpercentile(means, 97.5, axis=0)
        return ci_low.astype(np.float32), ci_high.astype(np.float32)

    # ------------------------------------------------------------------
    # Flush: produce PDFs + npz
    # ------------------------------------------------------------------

    def flush(
        self,
        output_dir: str,
        step: int,
        meta: Optional[dict] = None,
    ) -> None:
        """Compute per-stratum binned data, save .npz + render 4 per-stratum
        PDFs + 1 overlay PDF. Also save raw_trajectories.npz for offline use.
        """
        if not self.cfg.enabled:
            return
        if not self.rollouts:
            logger.warning("KL probe flush: no rollouts accumulated — skipping step %d", step)
            return

        os.makedirs(output_dir, exist_ok=True)
        meta = dict(meta or {})
        meta["step"] = step
        meta["n_total_rollouts"] = len(self.rollouts)

        # Per-stratum bin data — used for the 4 single-stratum plots AND
        # the overlay plot (one line per stratum).
        stratum_bins: dict[str, dict] = {}
        for stratum in STRATA:
            subset = self._stratum_rollouts(stratum)
            stratum_bins[stratum] = self._compute_bins(subset)

        # ---- 4 single-stratum plots + .npz ----
        for stratum, bins in stratum_bins.items():
            filename = f"step{step:03d}_{stratum}"
            npz_path = os.path.join(output_dir, f"{filename}.npz")
            pdf_path = os.path.join(output_dir, f"{filename}.pdf")

            bin_meta = {**meta, "stratum": stratum}
            np.savez(
                npz_path,
                bin_edges=bins["bin_edges"],
                bin_means=bins["bin_means"],
                bin_counts=bins["bin_counts"],
                survival=bins["survival"],
                bin_ci_low=bins["bin_ci_low"],
                bin_ci_high=bins["bin_ci_high"],
                n_rollouts=bins["n_rollouts"],
                meta=np.array(str(bin_meta), dtype=object),
            )
            try:
                plot_binned_pdf(
                    pdf_path=pdf_path,
                    bin_edges=bins["bin_edges"],
                    bin_means=bins["bin_means"],
                    bin_counts=bins["bin_counts"],
                    survival=bins["survival"],
                    bin_ci_low=bins["bin_ci_low"],
                    bin_ci_high=bins["bin_ci_high"],
                    n_rollouts=bins["n_rollouts"],
                    stratum=stratum,
                    min_tokens_per_bin=self.cfg.min_tokens_per_bin,
                    meta=bin_meta,
                )
            except Exception as e:
                logger.exception("KL probe: failed to render %s: %s", pdf_path, e)

        # ---- Overlay plot (4 stratum curves on one axes) ----
        overlay_pdf = os.path.join(output_dir, f"step{step:03d}_overlay.pdf")
        try:
            plot_overlay_pdf(
                pdf_path=overlay_pdf,
                stratum_bins=stratum_bins,
                min_tokens_per_bin=self.cfg.min_tokens_per_bin,
                meta=meta,
            )
        except Exception as e:
            logger.exception("KL probe: failed to render %s: %s", overlay_pdf, e)

        # Raw trajectory .npz — always save; cap at raw_exemplar_count from most
        # recent batch. Kept around for ad-hoc offline analysis even though we
        # no longer render per-rollout PDFs.
        exemplars = self._latest_batch[: self.cfg.raw_exemplar_count]
        raw_npz_path = os.path.join(output_dir, f"step{step:03d}_raw_trajectories.npz")
        _save_raw_trajectories_npz(raw_npz_path, exemplars, meta)


# ---------------------------------------------------------------------------
# Plotting (matplotlib). Lazy import so training without plotting still works.
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_binned_pdf(
    pdf_path: str,
    bin_edges: np.ndarray,
    bin_means: np.ndarray,
    bin_counts: np.ndarray,
    survival: np.ndarray,
    bin_ci_low: np.ndarray,
    bin_ci_high: np.ndarray,
    n_rollouts: int,
    stratum: str,
    min_tokens_per_bin: int,
    meta: dict,
) -> None:
    """Render one binned KL-by-position PDF for a single stratum.

    x-axis: token position (bin center).
    y-axis: mean reverse KL (nats).
    Shaded band: 95% clustered-bootstrap CI.
    Twin-axis: n_rollouts_surviving(t) — late-position bins backed by few
    rollouts are visually obvious.
    Sparse cells (count < min_tokens_per_bin) rendered in low-alpha markers.
    """
    plt = _mpl()

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    color = STRATUM_COLORS.get(stratum, "#333333")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    good = bin_counts >= min_tokens_per_bin
    sparse = (bin_counts > 0) & (~good)

    if good.any():
        gx = centers[good]
        gy = bin_means[good]
        glo = bin_ci_low[good]
        ghi = bin_ci_high[good]
        ax.plot(gx, gy, color=color, linewidth=1.6, label=stratum)
        mask = ~(np.isnan(glo) | np.isnan(ghi))
        if mask.any():
            ax.fill_between(
                gx[mask], glo[mask], ghi[mask], color=color, alpha=0.18, linewidth=0
            )

    if sparse.any():
        ax.scatter(
            centers[sparse],
            bin_means[sparse],
            color=color,
            alpha=0.25,
            s=10,
            marker="x",
            label="sparse (<min_tokens)",
        )

    ax.set_xlabel("token position")
    ax.set_ylabel("mean reverse KL (nats)")
    ax.set_xlim(0, bin_edges[-1])

    # Twin axis: rollout survival — flags late-position bins backed by few rollouts.
    if survival.size > 0:
        ax2 = ax.twinx()
        ax2.plot(centers, survival, color="#333333", linewidth=0.9, linestyle="--", alpha=0.5)
        ax2.set_ylabel("n rollouts surviving", color="#555555")
        ax2.tick_params(axis="y", colors="#555555")
        ax2.set_ylim(bottom=0)

    step = meta.get("step", "?")
    ctx_mode = meta.get("teacher_ctx_mode", "?")
    ax.set_title(f"{ctx_mode}  |  step {step}  |  {stratum}  |  n_rollouts={n_rollouts}")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overlay_pdf(
    pdf_path: str,
    stratum_bins: dict,
    min_tokens_per_bin: int,
    meta: dict,
) -> None:
    """Render the overlay plot: one curve per stratum on a single axes.

    Skips strata with no non-sparse bins. No CI bands (kept clean for the
    cross-stratum comparison). Twin-axis shows survival for the ``all``
    stratum (the other strata's survival curves aren't plotted to keep the
    plot readable).
    """
    plt = _mpl()

    fig, ax = plt.subplots(figsize=(9, 5))

    # First pass: plot each stratum's line. Use STRATA ordering for stable z-order.
    any_plotted = False
    for stratum in STRATA:
        bins = stratum_bins.get(stratum)
        if bins is None:
            continue
        centers = 0.5 * (bins["bin_edges"][:-1] + bins["bin_edges"][1:])
        good = bins["bin_counts"] >= min_tokens_per_bin
        if not good.any():
            continue
        color = STRATUM_COLORS.get(stratum, "#333333")
        label = f"{stratum}  (n={bins['n_rollouts']})"
        ax.plot(
            centers[good], bins["bin_means"][good],
            color=color, linewidth=1.6, label=label,
        )
        any_plotted = True

    ax.set_xlabel("token position")
    ax.set_ylabel("mean reverse KL (nats)")

    # Use the "all" stratum's bin edges for the x-limit (same across strata).
    any_bins = next(iter(stratum_bins.values()), None)
    if any_bins is not None:
        ax.set_xlim(0, any_bins["bin_edges"][-1])

        # Twin axis: survival for the "all" stratum only (cleanest summary of
        # how many rollouts reach each position).
        all_bins = stratum_bins.get("all")
        if all_bins is not None and all_bins["survival"].size > 0:
            centers = 0.5 * (all_bins["bin_edges"][:-1] + all_bins["bin_edges"][1:])
            ax2 = ax.twinx()
            ax2.plot(
                centers, all_bins["survival"],
                color="#333333", linewidth=0.9, linestyle="--", alpha=0.5,
                label="rollouts surviving (all)",
            )
            ax2.set_ylabel("n rollouts surviving (all)", color="#555555")
            ax2.tick_params(axis="y", colors="#555555")
            ax2.set_ylim(bottom=0)

    step = meta.get("step", "?")
    ctx_mode = meta.get("teacher_ctx_mode", "?")
    n_total = meta.get("n_total_rollouts", "?")
    ax.set_title(f"{ctx_mode}  |  step {step}  |  overlay (all strata)  |  total rollouts={n_total}")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    if any_plotted:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------


def _save_raw_trajectories_npz(path: str, exemplars: list[RolloutRecord], meta: dict) -> None:
    """Save raw per-rollout KL arrays (variable length) + metadata."""
    if not exemplars:
        np.savez(path, kl=np.array([], dtype=object), length=np.zeros(0, dtype=np.int64),
                 correct=np.zeros(0, dtype=bool), truncated=np.zeros(0, dtype=bool),
                 meta=np.array(str(meta), dtype=object))
        return
    lengths = np.array([r.length for r in exemplars], dtype=np.int64)
    correct = np.array([r.correct for r in exemplars], dtype=bool)
    truncated = np.array([r.truncated for r in exemplars], dtype=bool)
    # Store as object array since lengths vary.
    kl_arrays = np.empty(len(exemplars), dtype=object)
    for i, r in enumerate(exemplars):
        kl_arrays[i] = r.kl.astype(np.float32)
    np.savez(
        path,
        kl=kl_arrays,
        length=lengths,
        correct=correct,
        truncated=truncated,
        meta=np.array(str(meta), dtype=object),
        allow_pickle=True,
    )


def _parse_npz_meta(data) -> dict:
    """Parse the stringified meta dict stored in .npz files."""
    try:
        import ast
        meta_str = str(data["meta"])
        meta = ast.literal_eval(meta_str)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def plot_from_npz(npz_path: str, pdf_path: str, min_tokens_per_bin: int = 30) -> None:
    """Offline regeneration of a single-stratum binned PDF from a saved .npz.

    Used by ``replot_kl_probe.py`` to rebuild PDFs without re-running training.
    """
    data = np.load(npz_path, allow_pickle=True)
    meta = _parse_npz_meta(data)
    stratum = meta.get(
        "stratum",
        os.path.basename(npz_path).split("_", 1)[-1].replace(".npz", ""),
    )
    plot_binned_pdf(
        pdf_path=pdf_path,
        bin_edges=data["bin_edges"],
        bin_means=data["bin_means"],
        bin_counts=data["bin_counts"],
        survival=data["survival"],
        bin_ci_low=data["bin_ci_low"],
        bin_ci_high=data["bin_ci_high"],
        n_rollouts=int(data["n_rollouts"]),
        stratum=stratum,
        min_tokens_per_bin=min_tokens_per_bin,
        meta=meta,
    )


def overlay_from_npz_dir(
    session_plots_dir: str,
    step: int,
    pdf_path: str,
    min_tokens_per_bin: int = 30,
) -> None:
    """Regenerate the overlay plot for a step by collecting each stratum's .npz."""
    stratum_bins = {}
    meta: dict = {}
    for stratum in STRATA:
        npz_path = os.path.join(session_plots_dir, f"step{step:03d}_{stratum}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        if not meta:
            meta = _parse_npz_meta(data)
        stratum_bins[stratum] = {
            "bin_edges": data["bin_edges"],
            "bin_means": data["bin_means"],
            "bin_counts": data["bin_counts"],
            "survival": data["survival"],
            "bin_ci_low": data["bin_ci_low"],
            "bin_ci_high": data["bin_ci_high"],
            "n_rollouts": int(data["n_rollouts"]),
        }
    if not stratum_bins:
        return
    plot_overlay_pdf(
        pdf_path=pdf_path,
        stratum_bins=stratum_bins,
        min_tokens_per_bin=min_tokens_per_bin,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Sentence-boundary snap-back (shared with reinjection pipeline)
# ---------------------------------------------------------------------------


def snap_to_boundary(
    response_ids: list[int],
    target_pos: int,
    tokenizer,
    max_lookback: int = 200,
) -> int:
    """Find the last sentence-end before ``target_pos`` in ``response_ids``.

    Pragmatic decode→search→re-encode heuristic.

    CAVEAT: tokenizer round-trips are not exact bijections — decoding and
    re-encoding the same text can shift whitespace handling, merge/split
    tokens around punctuation, or add/remove leading spaces on BPE-style
    tokenizers. Treat the returned position as approximate; off by ±1 token
    is acceptable. The downstream pipeline operates on the actual returned
    index, so a small shift doesn't break correctness — it just means the
    snippet lands ±1 token from the ideal sentence boundary.

    Args:
        response_ids: the full response token ids.
        target_pos: nominal insertion target (e.g. k * interval).
        tokenizer: HuggingFace tokenizer.
        max_lookback: how far back (in tokens) to search for a boundary.

    Returns:
        Snapped position (<= target_pos). Falls back to ``target_pos`` if no
        boundary is found in the lookback window.
    """
    if target_pos <= 0:
        return target_pos
    start = max(0, target_pos - max_lookback)
    try:
        text = tokenizer.decode(response_ids[start:target_pos], skip_special_tokens=False)
    except Exception:
        return target_pos

    markers = (".\n", "!\n", "?\n", "\n\n", ". ", "! ", "? ")
    best = -1
    for marker in markers:
        idx = text.rfind(marker)
        if idx < 0:
            continue
        end = idx + len(marker)
        if end > best:
            best = end
    if best < 0:
        return target_pos
    snapped_text = text[:best]
    try:
        snap_tokens = tokenizer.encode(snapped_text, add_special_tokens=False)
    except Exception:
        return target_pos
    return min(start + len(snap_tokens), target_pos)


# ---------------------------------------------------------------------------
# Utility: find </think> close position in a response
# ---------------------------------------------------------------------------


_THINK_CLOSE_TEXT = "</think>"


def find_think_close(response_ids: list[int], tokenizer) -> Optional[int]:
    """Return the token-space position of the end of ``</think>`` in the response.

    Returns ``None`` if ``</think>`` is not present (truncated rollout).
    The returned position is the index of the first token AFTER ``</think>``
    in the response — reinjection is permitted only at positions < this.
    """
    try:
        text = tokenizer.decode(response_ids, skip_special_tokens=False)
    except Exception:
        return None
    char_idx = text.find(_THINK_CLOSE_TEXT)
    if char_idx < 0:
        return None
    close_end_char = char_idx + len(_THINK_CLOSE_TEXT)
    prefix_text = text[:close_end_char]
    try:
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    except Exception:
        return None
    return min(len(prefix_tokens), len(response_ids))
