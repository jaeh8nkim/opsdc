"""Invariant tests for teacher_ctx_reinjection and the kl_probe pipeline.

These tests are the gate that must pass before any reinjection experiment
run — silent alignment bugs in the teacher-input construction would
invalidate all downstream conclusions.

Focus:
  1. Equivalence test — with reinjection disabled OR empty positions, the
     teacher-side _tokenize_sequence output must be byte-identical to the
     baseline tokenization path.
  2. Snippet insertion — interleaving logic produces the expected teacher
     sequence and loss_mask.
  3. Cumulative-snippet invariant — the count of teacher response tokens
     (from teacher_loss_mask=1 positions in order) equals the student's
     response token count, even with multiple reinjections. This is what
     replaces the teacher_pos_map invariant in the original plan: because we
     set teacher_loss_mask=0 at snippet tokens, _forward_logits_* naturally
     yields a (N_student_response, V) tensor aligned 1:1 with the student side.
  4. <think> freeze — reinjection positions never exceed think_close_pos.
  5. Snap-back correctness — snapped positions always <= target, and are at
     valid sentence-end boundaries when available.

Run:
    python -m pytest workspace/src/self_distill_hybrid/test_reinjection_invariants.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch

from self_distill_hybrid.kl_probe import find_think_close, snap_to_boundary
from self_distill_hybrid.sd_verifier import (
    _tokenize_sequence,
    _tokenize_segment,
    _tokenize_response_with_eos,
)


# ---------------------------------------------------------------------------
# A small stub tokenizer that we fully control — byte-for-byte predictable.
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Whitespace-splitting tokenizer with integer IDs.

    Vocabulary is assigned on-demand: the first unique token seen gets id 0,
    next 1, etc. Good enough for invariant tests since we control all inputs.
    Reserves ids 1000+ for chat-template special tokens.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self.eos_token_id = 999
        self.pad_token_id = 998
        self._chat_template_tokens = [1000, 1001]  # stand-in for chat-template prefix
        self._gen_prompt_tokens = [1002]  # stand-in for "add_generation_prompt"

    def _ensure(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab) + 1  # 1..; leave 0 for padding-safety
        return self._vocab[tok]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        # Simple whitespace-separated tokens; preserve punctuation as its own token.
        # Use a deterministic split that is round-trippable: tokens are whitespace-separated
        # atoms, each mapped to an id. Punctuation stays attached to its adjacent word for
        # stability — not the cleanest, but enough for tests.
        for word in text.split(" "):
            if not word:
                continue
            ids.append(self._ensure(word))
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        tokens = [inv.get(int(i), "<unk>") for i in ids]
        return " ".join(tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        # Emit a short deterministic prefix based on content.
        combined = " ".join(m["content"] for m in messages)
        content_ids = self.encode(combined, add_special_tokens=False)
        return self._chat_template_tokens + content_ids + (
            self._gen_prompt_tokens if add_generation_prompt else []
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return StubTokenizer()


@pytest.fixture
def simple_prompt():
    return json.dumps([{"role": "user", "content": "what is 2 + 2 ?"}])


@pytest.fixture
def simple_response():
    # A response with a visible </think> boundary the find_think_close heuristic
    # can pick up via substring decode.
    return "<think> 2 + 2 is 4 . </think> Answer: 4 ."


@pytest.fixture
def snippet_ids(tokenizer):
    return tokenizer.encode(
        " Actually , I recall : key insight here . Continuing from this . ",
        add_special_tokens=False,
    )


# ---------------------------------------------------------------------------
# Test 1: equivalence — disabled reinjection == baseline path
# ---------------------------------------------------------------------------


def test_equivalence_empty_positions(tokenizer, simple_prompt, simple_response, snippet_ids):
    """Empty positions set → output identical to baseline."""
    baseline = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=256,
        pad_token_id=tokenizer.pad_token_id,
    )
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=snippet_ids,
        reinject_positions=set(),  # empty → no reinjection
    )
    assert baseline is not None and reinject is not None
    for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
        assert torch.equal(baseline[key], reinject[key]), f"mismatch in {key}"


def test_equivalence_none_snippet(tokenizer, simple_prompt, simple_response):
    """None snippet → output identical to baseline."""
    baseline = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=256,
        pad_token_id=tokenizer.pad_token_id,
    )
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=None,
        reinject_positions={1, 3},  # positions provided, but snippet None
    )
    assert baseline is not None and reinject is not None
    for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
        assert torch.equal(baseline[key], reinject[key]), f"mismatch in {key}"


# ---------------------------------------------------------------------------
# Test 2: snippet insertion — body length and loss_mask correct
# ---------------------------------------------------------------------------


def _response_len_with_eos(tokenizer, response_text):
    ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not ids or ids[-1] != tokenizer.eos_token_id:
        ids = ids + [tokenizer.eos_token_id]
    return len(ids)


def test_snippet_insertion_increases_teacher_length_by_snippet_len(
    tokenizer, simple_prompt, simple_response, snippet_ids,
):
    """Teacher input length = prompt + response + k * snippet, where k = len(positions)."""
    baseline = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id,
    )
    positions = {2, 5}
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=snippet_ids, reinject_positions=positions,
    )
    assert baseline is not None and reinject is not None

    baseline_nonpad = int(baseline["attention_mask"].sum())
    reinject_nonpad = int(reinject["attention_mask"].sum())

    assert reinject_nonpad - baseline_nonpad == len(positions) * len(snippet_ids), (
        f"expected +{len(positions) * len(snippet_ids)} tokens, got "
        f"+{reinject_nonpad - baseline_nonpad}"
    )


def test_response_token_count_preserved(
    tokenizer, simple_prompt, simple_response, snippet_ids,
):
    """Crucial invariant (replaces teacher_pos_map): teacher_loss_mask sums to
    exactly len(student response) regardless of reinjection. This is what makes
    _forward_logits_* naturally yield a (N_student_response, V) tensor aligned
    1:1 with the student side, without any separate index map.
    """
    positions = {1, 4, 7}
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=snippet_ids, reinject_positions=positions,
    )
    assert reinject is not None
    response_len = _response_len_with_eos(tokenizer, simple_response)
    teacher_response_token_count = int(reinject["loss_mask"].sum())
    assert teacher_response_token_count == response_len, (
        f"teacher loss_mask should sum to {response_len} (student response length); "
        f"got {teacher_response_token_count}"
    )


def test_loss_mask_zero_at_snippet_positions(
    tokenizer, simple_prompt, simple_response, snippet_ids,
):
    """Snippet tokens must have loss_mask=0 so they don't flow into the KL loss."""
    positions = {2}
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=snippet_ids, reinject_positions=positions,
    )
    assert reinject is not None
    input_ids = reinject["input_ids"].tolist()
    loss_mask = reinject["loss_mask"].tolist()

    # Find the first snippet occurrence: longest contiguous match of snippet_ids.
    found = False
    for i in range(len(input_ids) - len(snippet_ids) + 1):
        if input_ids[i : i + len(snippet_ids)] == snippet_ids:
            assert all(loss_mask[j] == 0 for j in range(i, i + len(snippet_ids))), (
                "loss_mask must be 0 at all snippet token positions"
            )
            found = True
            break
    assert found, "expected snippet to appear in the teacher input_ids"


# ---------------------------------------------------------------------------
# Test 3: <think> freeze — reinjection never fires past think_close
# ---------------------------------------------------------------------------


def test_find_think_close_locates_boundary(tokenizer, simple_response):
    """<think> close position is correctly identified in the tokenized response."""
    response_ids = tokenizer.encode(simple_response, add_special_tokens=False)
    close_pos = find_think_close(response_ids, tokenizer)
    assert close_pos is not None
    # Decode the prefix up to close_pos and verify it contains </think>.
    decoded_prefix = tokenizer.decode(response_ids[:close_pos])
    assert "</think>" in decoded_prefix, (
        f"decoded prefix up to close_pos should contain </think>; got {decoded_prefix!r}"
    )


def test_find_think_close_returns_none_for_truncated():
    """Response without </think> → None (truncated)."""
    tok = StubTokenizer()
    ids = tok.encode("<think> wandering forever .", add_special_tokens=False)
    assert find_think_close(ids, tok) is None


# ---------------------------------------------------------------------------
# Test 4: snap-back correctness
# ---------------------------------------------------------------------------


def test_snap_to_boundary_bounded_above_by_target(tokenizer):
    """Snap-back never overshoots the target position (clamp guard)."""
    text = "first sentence . second sentence . third sentence ."
    ids = tokenizer.encode(text, add_special_tokens=False)
    for target in [1, 2, 3, 5, len(ids) - 1]:
        snapped = snap_to_boundary(ids, target, tokenizer, max_lookback=100)
        assert snapped <= target, f"snap({target}) = {snapped} > target"


def test_snap_to_boundary_fallback_on_no_boundary(tokenizer):
    """With no boundary in lookback, snap returns target unchanged."""
    ids = tokenizer.encode("a b c d e", add_special_tokens=False)
    snapped = snap_to_boundary(ids, 3, tokenizer, max_lookback=100)
    assert snapped == 3


def test_snap_to_boundary_zero_target(tokenizer):
    """target=0 → returns 0 (trivial case, no search performed)."""
    ids = tokenizer.encode("any tokens here .", add_special_tokens=False)
    assert snap_to_boundary(ids, 0, tokenizer) == 0


# ---------------------------------------------------------------------------
# Test 5: multi-reinjection body structure
# ---------------------------------------------------------------------------


def test_multi_reinjection_ordering(tokenizer, simple_prompt, simple_response, snippet_ids):
    """With k reinjections at increasing positions, snippets appear in the
    teacher body in the same order, each separating the correct student tokens.
    """
    positions = sorted({1, 3, 5})
    reinject = _tokenize_sequence(
        simple_prompt, simple_response, tokenizer, max_length=1024,
        pad_token_id=tokenizer.pad_token_id,
        reinject_snippet_ids=snippet_ids,
        reinject_positions=set(positions),
    )
    assert reinject is not None
    input_ids = reinject["input_ids"].tolist()

    snippet_count = 0
    i = 0
    while i < len(input_ids) - len(snippet_ids) + 1:
        if input_ids[i : i + len(snippet_ids)] == snippet_ids:
            snippet_count += 1
            i += len(snippet_ids)
        else:
            i += 1
    assert snippet_count == len(positions), (
        f"expected {len(positions)} snippet occurrences, got {snippet_count}"
    )


# ---------------------------------------------------------------------------
# Test 6: multi-pass segment invariants
# ---------------------------------------------------------------------------


def test_multipass_segment_coverage_disjoint(tokenizer, simple_prompt, simple_response, snippet_ids):
    """For a sample with K reinjections, the K+1 student-segment loss_masks
    must partition exactly the full response (non-overlapping and covering).
    """
    response_ids = _tokenize_response_with_eos(simple_response, tokenizer)
    L = len(response_ids)
    positions = sorted([2, 4])
    boundaries = [0] + positions + [L]

    covered = set()
    for k in range(len(boundaries) - 1):
        s_row = _tokenize_segment(
            simple_prompt, response_ids, boundaries[k], boundaries[k + 1],
            snippet_ids=None, tokenizer=tokenizer, max_length=512,
            pad_token_id=tokenizer.pad_token_id, is_teacher=False,
        )
        assert s_row is not None
        # Loss-mask=1 count should equal segment length.
        n_marked = int(s_row["loss_mask"].sum())
        expected = boundaries[k + 1] - boundaries[k]
        assert n_marked == expected, f"segment {k}: {n_marked} != {expected}"

        # Positions in input_ids that carry loss_mask=1 should be disjoint
        # across segments (different k).
        idxs = set(s_row["loss_mask"].nonzero(as_tuple=True)[0].tolist())
        assert not (idxs & covered), f"segment {k} overlaps earlier segments"
        covered |= idxs

    # Total count covers exactly the response.
    assert len(covered) == L


def test_multipass_teacher_segment_loss_mask_count(
    tokenizer, simple_prompt, simple_response, snippet_ids,
):
    """Teacher segment's loss_mask=1 count == segment length (even with snippet)."""
    response_ids = _tokenize_response_with_eos(simple_response, tokenizer)
    L = len(response_ids)

    # Segment 1 (with snippet): response[3..6]
    t_row = _tokenize_segment(
        simple_prompt, response_ids, segment_start=3, segment_end=6,
        snippet_ids=snippet_ids, tokenizer=tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id, is_teacher=True,
    )
    assert t_row is not None
    assert int(t_row["loss_mask"].sum()) == 3, "segment length 3 → 3 loss_mask=1 positions"

    # Segment 0 (no snippet): response[0..3]
    t_row0 = _tokenize_segment(
        simple_prompt, response_ids, segment_start=0, segment_end=3,
        snippet_ids=None, tokenizer=tokenizer, max_length=512,
        pad_token_id=tokenizer.pad_token_id, is_teacher=True,
    )
    assert t_row0 is not None
    assert int(t_row0["loss_mask"].sum()) == 3


def test_multipass_teacher_segment_has_no_prior_snippets(
    tokenizer, simple_prompt, simple_response, snippet_ids,
):
    """Crucial replacement-semantics check: segment k's teacher input has the
    current snippet right before segment start, but NO prior snippets in it.
    We verify by counting snippet token sequences in the teacher input.
    """
    response_ids = _tokenize_response_with_eos(simple_response, tokenizer)

    # Build segment 2 (imagine reinject at positions 2 and 4 → segment 2 covers
    # response[4..end], current snippet is the segment-2 snippet).
    t_row = _tokenize_segment(
        simple_prompt, response_ids, segment_start=4, segment_end=len(response_ids),
        snippet_ids=snippet_ids, tokenizer=tokenizer, max_length=1024,
        pad_token_id=tokenizer.pad_token_id, is_teacher=True,
    )
    assert t_row is not None
    input_ids = t_row["input_ids"].tolist()

    # Count how many times snippet_ids appears contiguously.
    n_occ = 0
    i = 0
    while i <= len(input_ids) - len(snippet_ids):
        if input_ids[i : i + len(snippet_ids)] == snippet_ids:
            n_occ += 1
            i += len(snippet_ids)
        else:
            i += 1
    assert n_occ == 1, f"expected 1 snippet (current only), got {n_occ}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
