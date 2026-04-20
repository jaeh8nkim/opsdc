"""Plumbing tests for correctness_branched_kl.

Covers the sample-direction helper, padded-mask build, round-trip via
``_extract_response_values``, multi-pass row replication pattern, cumulative
1:1 assertion defense, and ``validate_opsd_config`` decisions. These tests
avoid importing ``opsd_trainer`` / ``opsd_worker`` directly because those
modules pull in Ray / VERL / torchdata; we exercise the lightweight helpers
that can be imported as pure Python.
"""
import random
import sys
from pathlib import Path

import pytest
import torch

# Ensure the package directory is importable without pulling VERL in. We
# import the standalone utility module and the static helpers via fully
# qualified dotted paths where needed.
_PKG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG_DIR.parent))

from self_distill_hybrid.opsd_config_validation import (  # noqa: E402
    VALID_TRUNCATED_HANDLING,
    compute_use_reverse_kl_sample,
    validate_opsd_config,
)


# ---------------------------------------------------------------------------
# Inlined copies of the trainer/worker helpers under test.
# Keeping them inline preserves the test-isolation policy established by
# test_opsd_jsd.py (which copies _JSD helpers to avoid VERL imports).
# ---------------------------------------------------------------------------


def _build_use_reverse_kl_mask_padded(
    student_loss_mask: torch.Tensor,
    row_use_reverse_kl: list[bool],
) -> torch.Tensor:
    """Mirror of OPSDTrainer._build_use_reverse_kl_mask_padded."""
    B, max_L = student_loss_mask.shape
    assert len(row_use_reverse_kl) == B
    out = torch.zeros(B, max_L, dtype=torch.bool)
    bool_mask = student_loss_mask.bool()
    for i, use_rev in enumerate(row_use_reverse_kl):
        if use_rev:
            out[i] = bool_mask[i]
    return out


def _extract_response_values(
    tensor_padded: torch.Tensor, loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Mirror of OPSDWorker._extract_response_values."""
    shift_v = tensor_padded[:, 1:]
    shift_mask = loss_mask[:, 1:]
    B, S = shift_v.shape
    flat_v = shift_v.reshape(B * S)
    flat_mask = shift_mask.reshape(B * S)
    response_indices = flat_mask.nonzero(as_tuple=True)[0]
    return flat_v[response_indices]


# ---------------------------------------------------------------------------
# compute_use_reverse_kl_sample: 4-way truth table + property tests
# ---------------------------------------------------------------------------


class TestComputeUseReverseKLSample:
    """Core precedence rule: correct -> reverse (always)."""

    @pytest.fixture
    def correct(self):
        # (T,F), (F,F), (T,T), (F,T) in that order
        return [True, False, True, False]

    @pytest.fixture
    def truncated(self):
        return [False, False, True, True]

    def test_as_correct_expected(self, correct, truncated):
        out = compute_use_reverse_kl_sample(correct, truncated, "as_correct")
        assert out == [True, False, True, True]

    def test_as_incorrect_expected(self, correct, truncated):
        out = compute_use_reverse_kl_sample(correct, truncated, "as_incorrect")
        assert out == [True, False, True, False]

    def test_correct_and_truncated_always_reverse(self, correct, truncated):
        """Proposal 1 invariant: (correct=T, truncated=T) -> reverse in BOTH modes.

        This is the primary regression guard for the "correct -> reverse"
        invariant. If someone flips it back to `correct AND NOT truncated`,
        this test will fail loudly.
        """
        for mode in VALID_TRUNCATED_HANDLING:
            out = compute_use_reverse_kl_sample(correct, truncated, mode)
            assert out[2] is True, (
                f"(correct=T, truncated=T) must route to reverse KL in "
                f"mode={mode!r}; got forward."
            )

    def test_property_correct_always_reverse(self):
        """Randomized property test: wherever correct[i] is True, the output
        must be True, regardless of truncated[i] or the mode."""
        rng = random.Random(12345)
        for trial in range(100):
            n = rng.randint(1, 32)
            c = [rng.random() < 0.5 for _ in range(n)]
            t = [rng.random() < 0.5 for _ in range(n)]
            for mode in VALID_TRUNCATED_HANDLING:
                out = compute_use_reverse_kl_sample(c, t, mode)
                for i in range(n):
                    if c[i]:
                        assert out[i] is True, (
                            f"trial={trial} i={i} mode={mode}: "
                            f"correct=True but got forward KL"
                        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_use_reverse_kl_sample([True, False], [True], "as_correct")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            compute_use_reverse_kl_sample([True], [False], "nonsense")


# ---------------------------------------------------------------------------
# Padded mask build + extract round-trip
# ---------------------------------------------------------------------------


class TestPaddedMaskBuild:
    def test_response_positions_get_direction(self):
        # 3 rows, max_L=8. Response positions (loss_mask=1) vary per row.
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0, 0],  # response at 2..4
            [0, 1, 1, 1, 1, 1, 0, 0],  # response at 1..5
            [0, 0, 0, 1, 1, 1, 1, 1],  # response at 3..7
        ], dtype=torch.float32)
        row_use_rev = [True, False, True]

        padded = _build_use_reverse_kl_mask_padded(loss_mask, row_use_rev)

        assert padded.dtype == torch.bool
        # Row 0 (reverse): response positions True, non-response False.
        assert padded[0].tolist() == [False, False, True, True, True, False, False, False]
        # Row 1 (forward): entire row False (including response positions).
        assert padded[1].tolist() == [False] * 8
        # Row 2 (reverse): response positions True.
        assert padded[2].tolist() == [False, False, False, True, True, True, True, True]

    def test_non_response_stays_false_on_reverse_row(self):
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        padded = _build_use_reverse_kl_mask_padded(loss_mask, [True])
        # Non-response positions must be False even when the row is reverse-KL.
        for pos in [0, 1, 5, 6, 7]:
            assert padded[0, pos].item() is False

    def test_length_mismatch_raises(self):
        loss_mask = torch.zeros(3, 4)
        with pytest.raises(AssertionError):
            _build_use_reverse_kl_mask_padded(loss_mask, [True, False])


class TestExtractRoundtrip:
    """Flatten+shift behavior on per-token masks (mirrors worker)."""

    def test_bool_roundtrip(self):
        # Row 0 response at positions 2..4 (loss_mask shift => columns 1..3 of shifted).
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0],
        ], dtype=torch.float32)
        row_use_rev = [True, False]
        padded = _build_use_reverse_kl_mask_padded(loss_mask, row_use_rev)

        flat = _extract_response_values(padded, loss_mask)

        # Row 0: response at cols 2,3,4. After shift: mask-1 at cols 1,2,3
        # of shift_mask (i.e., original cols 2,3,4). All True because row is
        # reverse-KL.
        # Row 1: response at cols 1,2. After shift: shift_mask True at cols
        # 0,1 (original 1,2). All False because row is forward-KL.
        assert flat.dtype == torch.bool
        assert flat.tolist() == [True, True, True, False, False]


# ---------------------------------------------------------------------------
# Multi-pass row replication pattern
# ---------------------------------------------------------------------------


class TestMultipassRowReplication:
    def test_replication_from_shard_batch_idx(self):
        """Replicate sample-level direction to rows using shard_batch_idx.

        Mirrors the inner logic of `_opsd_update`'s multipass branch:
        ``row_use_reverse_kl = [use_reverse_kl_sample[s] for s in row_sample_idx]``.
        """
        use_reverse_kl_sample = [True, False, True]  # 3 active samples
        row_sample_idx = [0, 0, 1, 2, 2]  # 5 rows: sample-0 twice, sample-1 once, sample-2 twice

        row_use_reverse_kl = [use_reverse_kl_sample[s] for s in row_sample_idx]

        assert row_use_reverse_kl == [True, True, False, True, True]


# ---------------------------------------------------------------------------
# Cumulative 1:1 assertion defense
# ---------------------------------------------------------------------------


class TestCumulativeAssertion:
    """Pre-existing silent drop in build_opsd_batch is not fixed; branched KL
    must assert loudly rather than silently misalign its direction mask."""

    def test_len_mismatch_trips_assertion(self):
        """Directly exercise the assertion shape used inside _opsd_update."""
        use_reverse_kl_sample = [True, False, True, False]  # 4 active
        n_rows = 3  # builder silently dropped 1

        with pytest.raises(AssertionError):
            assert n_rows == len(use_reverse_kl_sample), (
                f"build_opsd_batch dropped {len(use_reverse_kl_sample) - n_rows} "
                "sample(s); branched KL relies on 1:1 row-to-active mapping."
            )


# ---------------------------------------------------------------------------
# validate_opsd_config
# ---------------------------------------------------------------------------


class TestValidateOpsdConfig:
    def test_default_ok(self):
        lt, kg, th = validate_opsd_config({})
        assert lt == "jsd"
        assert kg == "all"
        assert th is None

    def test_reverse_kl_ok(self):
        lt, kg, th = validate_opsd_config({"loss_type": "reverse_kl"})
        assert lt == "reverse_kl"
        assert kg == "all"
        assert th is None

    def test_branched_requires_truncated_handling(self):
        with pytest.raises(ValueError, match="truncated_handling"):
            validate_opsd_config({"loss_type": "correctness_branched_kl"})

    def test_branched_rejects_invalid_truncated_handling(self):
        with pytest.raises(ValueError, match="truncated_handling"):
            validate_opsd_config({
                "loss_type": "correctness_branched_kl",
                "truncated_handling": "bogus",
            })

    def test_branched_requires_kl_gating_all(self):
        with pytest.raises(ValueError, match="kl_gating"):
            validate_opsd_config({
                "loss_type": "correctness_branched_kl",
                "kl_gating": "correct_only",
                "truncated_handling": "as_correct",
            })

    @pytest.mark.parametrize("th", ["as_correct", "as_incorrect"])
    def test_branched_ok_with_all_gating_and_handling(self, th):
        lt, kg, th_out = validate_opsd_config({
            "loss_type": "correctness_branched_kl",
            "kl_gating": "all",
            "truncated_handling": th,
        })
        assert lt == "correctness_branched_kl"
        assert kg == "all"
        assert th_out == th

    def test_invalid_loss_type(self):
        with pytest.raises(ValueError, match="loss_type"):
            validate_opsd_config({"loss_type": "nope"})

    def test_invalid_kl_gating(self):
        with pytest.raises(ValueError, match="kl_gating"):
            validate_opsd_config({"kl_gating": "nope"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
