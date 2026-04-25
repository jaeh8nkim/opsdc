"""Pure config validation for OPSD.

Intentionally avoids torch / VERL / Ray imports so config-only unit tests
can run without pulling the heavy training-side dependency tree. Keep this
module on the standard library only.
"""
from typing import Optional

VALID_LOSS_TYPES = ("jsd", "reverse_kl", "forward_kl", "correctness_branched_kl")
VALID_GATINGS = ("all", "correct_only", "incorrect_only", "correct_and_truncated")
VALID_TRUNCATED_HANDLING = ("as_correct", "as_incorrect")


def validate_opsd_config(opsd_config) -> tuple[str, str, Optional[str]]:
    """Validate OPSD config fields. Returns (loss_type, kl_gating, truncated_handling).

    For ``correctness_branched_kl``:
      - ``kl_gating`` must be ``"all"`` (branching needs both correct and incorrect
        samples; other gatings drop one side and make branching degenerate).
      - ``truncated_handling`` must be one of ``VALID_TRUNCATED_HANDLING`` (no default).
    """
    loss_type = opsd_config.get("loss_type", "jsd")
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(
            f"Invalid loss_type: {loss_type!r}. Expected one of {VALID_LOSS_TYPES}."
        )

    kl_gating = opsd_config.get("kl_gating", "all")
    if kl_gating not in VALID_GATINGS:
        raise ValueError(
            f"Invalid kl_gating: {kl_gating!r}. Expected one of {VALID_GATINGS}."
        )

    truncated_handling = opsd_config.get("truncated_handling", None)
    if loss_type == "correctness_branched_kl":
        if kl_gating != "all":
            raise ValueError(
                f"loss_type={loss_type!r} requires kl_gating='all' "
                "(direction branching needs both correct and incorrect samples; "
                "other gatings drop one side and make branching degenerate)."
            )
        if truncated_handling not in VALID_TRUNCATED_HANDLING:
            raise ValueError(
                f"loss_type={loss_type!r} requires truncated_handling to be "
                f"one of {VALID_TRUNCATED_HANDLING} (got {truncated_handling!r})."
            )
    return loss_type, kl_gating, truncated_handling


def compute_use_reverse_kl_sample(
    active_correct: list[bool],
    active_truncated: list[bool],
    truncated_handling: str,
) -> list[bool]:
    """Per-sample direction decision with correct-wins precedence.

    Proposal 1 invariant: ``correct=True`` -> reverse KL, regardless of truncated.
    ``truncated_handling`` only resolves the ``(correct=False, truncated=True)`` case.

    Truth table:
      (T, F) -> reverse     (correct wins)
      (T, T) -> reverse     (correct wins)
      (F, F) -> forward     (genuine wrong-reasoning)
      (F, T) -> depends on ``truncated_handling``
    """
    if len(active_correct) != len(active_truncated):
        raise ValueError(
            f"active_correct ({len(active_correct)}) and active_truncated "
            f"({len(active_truncated)}) length mismatch"
        )
    if truncated_handling == "as_correct":
        return [c or t for c, t in zip(active_correct, active_truncated)]
    elif truncated_handling == "as_incorrect":
        return list(active_correct)
    else:
        raise ValueError(
            f"Unknown truncated_handling: {truncated_handling!r}. "
            f"Expected one of {VALID_TRUNCATED_HANDLING}."
        )
