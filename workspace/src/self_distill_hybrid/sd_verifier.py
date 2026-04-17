"""
Response verification and SFT batch construction for self-distillation.

Two responsibilities:
1. Verify student-generated responses (structure + math correctness)
2. Build tokenized SFT training batches from verified (correct) responses
"""

import json
import logging
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def validate_response_structure(text: str) -> bool:
    """Check that a student response has the required <think>…</think> structure.

    Accepts either ``Answer: X`` (Qwen2.5-style) or ``\\boxed{X}`` (Qwen3-style)
    as the answer indicator.  Mirrors the logic from
    ``self_distill_gen.py._validate_response_structure``.
    """
    if not text or len(text.strip()) < 50:
        return False

    has_think_open = "<think>" in text
    has_think_close = "</think>" in text
    has_answer = "answer:" in text.lower() or "\\boxed{" in text

    if not (has_think_open and has_think_close and has_answer):
        return False

    # The <think> block must contain substantive content
    try:
        think_start = text.index("<think>") + len("<think>")
        think_end = text.index("</think>")
        think_content = text[think_start:think_end].strip()
        if len(think_content) < 20:
            return False
    except ValueError:
        return False

    return True


def verify_response(
    response_text: str,
    ground_truth: str,
    check_structure: bool = True,
) -> tuple[bool, str]:
    """Verify a single student response for correctness and structure.

    Tries ``Answer:`` extraction first (matches prompt instruction); if that
    yields ``[INVALID]``, falls back to ``\\boxed{}`` extraction (Qwen3 thinking
    mode ignores the Answer: instruction and outputs \\boxed{} only).

    Args:
        response_text: The student's generated response.
        ground_truth: Expected answer string.
        check_structure: Whether to also require <think> + answer structure.

    Returns:
        (is_correct, extracted_prediction)
    """
    from verl.utils.reward_score.math_dapo import (
        last_boxed_only_string,
        normalize_final_answer,
        remove_boxed,
        verify,
    )

    if not response_text or not response_text.strip():
        return False, ""

    # Primary: "Answer:" extraction (matches prompt instruction)
    is_correct, pred = verify(response_text, ground_truth)

    # Fallback: \boxed{} extraction (Qwen3 thinking mode ignores Answer: instruction)
    if pred == "[INVALID]":
        boxed = last_boxed_only_string(response_text)
        if boxed is not None:
            pred = normalize_final_answer(remove_boxed(boxed))
            gt_norm = normalize_final_answer(ground_truth)
            is_correct = pred == gt_norm

    if is_correct and check_structure:
        if not validate_response_structure(response_text):
            return False, pred

    return is_correct, pred


def verify_batch(
    responses: list[str],
    ground_truths: list[str],
    check_structure: bool = True,
) -> tuple[list[bool], list[str]]:
    """Verify a batch of responses.

    Returns:
        (correct_mask, predictions) — both lists of length len(responses).
    """
    correct_mask = []
    predictions = []
    for resp, gt in zip(responses, ground_truths):
        ok, pred = verify_response(resp, gt, check_structure=check_structure)
        correct_mask.append(ok)
        predictions.append(pred)
    return correct_mask, predictions


# ---------------------------------------------------------------------------
# SFT Batch Construction
# ---------------------------------------------------------------------------


def build_sft_batch(
    sft_prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32768,
) -> Optional[DataProto]:
    """Build a tokenized SFT training batch from verified correct responses.

    For each (sft_prompt, response) pair:
      1. Apply chat template to sft_prompt to get prompt token IDs
      2. Tokenize response text (+ EOS)
      3. Concatenate and create loss_mask (0 for prompt, 1 for response)
      4. Right-pad to max_length

    Args:
        sft_prompts: List of JSON-string chat-format messages (SFT prompts).
        responses: List of verified response strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Max total sequence length (prompt + response).

    Returns:
        DataProto with batch keys: input_ids, attention_mask, position_ids, loss_mask.
        Returns None if no valid samples after filtering.
    """
    if not sft_prompts:
        return None

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_loss_mask = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    skipped = 0
    for sft_prompt_str, response_text in zip(sft_prompts, responses):
        # Parse prompt and apply chat template
        messages = json.loads(sft_prompt_str)
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )

        # Tokenize response (no special tokens — we add EOS manually)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)

        # Ensure EOS at the end
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        # Truncate if exceeds max_length
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            loss_mask = loss_mask[:max_length]
            # Make sure we don't lose the last few tokens that matter
            # If truncation cuts into response, that's ok — we still train on
            # the partial response up to max_length

        seq_len = len(full_ids)
        if seq_len < 2:
            skipped += 1
            continue

        # Right-pad to max_length
        pad_len = max_length - seq_len
        input_ids = full_ids + [pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len
        loss_mask_padded = loss_mask + [0] * pad_len
        position_ids = list(range(seq_len)) + [0] * pad_len

        all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        all_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
        all_position_ids.append(torch.tensor(position_ids, dtype=torch.long))
        all_loss_mask.append(torch.tensor(loss_mask_padded, dtype=torch.float32))

    if skipped:
        logger.warning("Skipped %d samples during SFT batch construction (too short)", skipped)

    if not all_input_ids:
        return None

    batch_dict = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "position_ids": torch.stack(all_position_ids),
        "loss_mask": torch.stack(all_loss_mask),
    }
    return DataProto.from_single_dict(batch_dict)


# ---------------------------------------------------------------------------
# OPSD Batch Construction
# ---------------------------------------------------------------------------


def _tokenize_sequence(
    prompt_str: str,
    response_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
    reinject_snippet_ids: Optional[list[int]] = None,
    reinject_positions: Optional[set[int]] = None,
) -> Optional[dict]:
    """Tokenize a single (prompt, response) pair into padded tensors.

    When ``reinject_snippet_ids`` and ``reinject_positions`` are provided,
    interleaves the snippet into the response at each student position in
    ``reinject_positions``. The loss_mask is 0 for snippet tokens (teacher
    context only, not scored) and 1 for student response tokens. The output
    ``_forward_logits_*`` path naturally produces logits for student response
    tokens in order, so no separate teacher_pos_map is needed — see plan.

    Returns dict with input_ids, attention_mask, position_ids, loss_mask,
    or None if the sequence is too short.
    """
    messages = json.loads(prompt_str)
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]

    if reinject_snippet_ids and reinject_positions:
        # Build interleaved response body + corresponding loss mask segment.
        body_ids: list[int] = []
        body_loss: list[int] = []
        for s_t, tok in enumerate(response_ids):
            if s_t in reinject_positions:
                body_ids.extend(reinject_snippet_ids)
                body_loss.extend([0] * len(reinject_snippet_ids))
            body_ids.append(tok)
            body_loss.append(1)
        full_ids = prompt_ids + body_ids
        loss_mask = [0] * len(prompt_ids) + body_loss
    else:
        full_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        loss_mask = loss_mask[:max_length]

    seq_len = len(full_ids)
    if seq_len < 2:
        return None

    pad_len = max_length - seq_len
    return {
        "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
        "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
        "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
    }


def _tokenize_segment(
    prompt_str: str,
    response_ids_with_eos: list[int],
    segment_start: int,
    segment_end: int,
    snippet_ids: Optional[list[int]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
    is_teacher: bool,
) -> Optional[dict]:
    """Tokenize one segment of one sample for multi-pass reinjection.

    For multi-pass, each sample is expanded into K+1 rows (one per segment).
    Each row gates the loss to ONLY the segment's response tokens via loss_mask,
    so the forward path naturally yields logits for just that segment.

    Args:
        prompt_str: JSON chat messages.
        response_ids_with_eos: fully tokenized response, EOS appended.
        segment_start: first response-token index that this segment covers (inclusive).
        segment_end: one-past-last response-token index (exclusive). Segment covers
            response[segment_start:segment_end].
        snippet_ids: teacher-only snippet to insert RIGHT BEFORE segment_start.
            Empty/None for segment 0 (no prior reinjection). Ignored when
            ``is_teacher=False`` (student never sees snippets).
        is_teacher: True → teacher frame (may contain snippet). False → student
            frame (full response unchanged, no snippet).

    Returns:
        dict with input_ids, attention_mask, position_ids, loss_mask;
        None if the sequence is too short.

    Shape invariant: the loss_mask marks exactly ``segment_end - segment_start``
    positions across the output (one per response token in this segment). The
    forward path extracts one logit per marked position, so both teacher and
    student rows of the same segment contribute the same count of response
    logits — aligned 1:1 across the pair.
    """
    messages = json.loads(prompt_str)
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    if is_teacher:
        # Teacher body: pre-segment response context + snippet + segment tokens.
        # Tokens AFTER segment_end are intentionally omitted — the teacher doesn't
        # need to produce logits for those in this row.
        pre = list(response_ids_with_eos[:segment_start])
        seg_tokens = list(response_ids_with_eos[segment_start:segment_end])
        snip = list(snippet_ids) if snippet_ids else []
        body_ids = pre + snip + seg_tokens
        body_loss = [0] * len(pre) + [0] * len(snip) + [1] * len(seg_tokens)
    else:
        # Student: full response, loss_mask=1 only at the segment's response tokens.
        body_ids = list(response_ids_with_eos)
        body_loss = [0] * len(body_ids)
        for p in range(segment_start, segment_end):
            body_loss[p] = 1

    full_ids = prompt_ids + body_ids
    loss_mask = [0] * len(prompt_ids) + body_loss

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        loss_mask = loss_mask[:max_length]

    seq_len = len(full_ids)
    if seq_len < 2:
        return None

    pad_len = max_length - seq_len
    return {
        "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
        "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
        "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
    }


def _tokenize_response_with_eos(
    response_text: str,
    tokenizer: PreTrainedTokenizer,
) -> list[int]:
    """Tokenize response and append EOS if missing (matches _tokenize_sequence)."""
    ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not ids or ids[-1] != tokenizer.eos_token_id:
        ids = ids + [tokenizer.eos_token_id]
    return ids


def build_opsd_batch_multipass(
    teacher_prompts: list[str],
    student_prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32768,
    reinject_snippets: Optional[list[Optional[list[int]]]] = None,
    reinject_positions: Optional[list[Optional[list[int]]]] = None,
) -> Optional[tuple[DataProto, list[int], list[int]]]:
    """Build expanded (per-segment) OPSD batch for multi-pass reinjection.

    Each sample with K reinjection positions is expanded into K+1 rows. Segment
    k covers response tokens [b_k .. b_{k+1} - 1] where b_0=0, b_k=r_k for
    k=1..K, b_{K+1}=L_response.

    Per row:
      - Teacher input = [prompt, ctx, <think>, response[0..b_k-1] +
        snippet_k (if k>0) + response[b_k..b_{k+1}-1]]. Note: no prior snippets
        remain in context — earlier reinjections are discarded (the
        replacement semantics).
      - Student input = [prompt_student, full response].
      - Both loss_masks = 1 only at response[b_k..b_{k+1}-1] positions.

    Returns:
        Tuple of (DataProto, shard_batch_idx, segment_idx) where:
          - DataProto has (N_expanded, max_L) tensors.
          - shard_batch_idx: list of length N_expanded, entry i = original
            sample index this row belongs to.
          - segment_idx: list of length N_expanded, entry i = segment number
            for this row within its sample (0..K).
        Or None if no valid samples.
    """
    if not teacher_prompts:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    n = len(teacher_prompts)
    if reinject_snippets is None:
        reinject_snippets = [None] * n
    if reinject_positions is None:
        reinject_positions = [None] * n

    teacher_seqs = []
    student_seqs = []
    shard_batch_idx: list[int] = []
    segment_idx: list[int] = []
    skipped_samples = 0

    for sample_i, (t_prompt, s_prompt, response_text, snippet_ids, positions) in enumerate(zip(
        teacher_prompts, student_prompts, responses,
        reinject_snippets, reinject_positions,
    )):
        response_ids = _tokenize_response_with_eos(response_text, tokenizer)
        L_resp = len(response_ids)
        if L_resp < 2:
            skipped_samples += 1
            continue

        # Segment boundaries: b_0=0, b_1..b_K=positions, b_{K+1}=L_resp.
        sorted_positions = sorted(int(p) for p in (positions or []) if 0 < int(p) < L_resp)
        boundaries = [0] + sorted_positions + [L_resp]

        sample_rows_teacher = []
        sample_rows_student = []
        sample_rows_meta = []
        sample_ok = True
        for k in range(len(boundaries) - 1):
            seg_start = boundaries[k]
            seg_end = boundaries[k + 1]
            if seg_end <= seg_start:
                continue  # skip degenerate empty segments
            snip_for_segment = snippet_ids if k > 0 and snippet_ids else None
            t_row = _tokenize_segment(
                t_prompt, response_ids, seg_start, seg_end, snip_for_segment,
                tokenizer, max_length, pad_token_id, is_teacher=True,
            )
            s_row = _tokenize_segment(
                s_prompt, response_ids, seg_start, seg_end, None,
                tokenizer, max_length, pad_token_id, is_teacher=False,
            )
            if t_row is None or s_row is None:
                sample_ok = False
                break
            sample_rows_teacher.append(t_row)
            sample_rows_student.append(s_row)
            sample_rows_meta.append(k)

        if not sample_ok or not sample_rows_teacher:
            skipped_samples += 1
            continue

        teacher_seqs.extend(sample_rows_teacher)
        student_seqs.extend(sample_rows_student)
        shard_batch_idx.extend([sample_i] * len(sample_rows_teacher))
        segment_idx.extend(sample_rows_meta)

    if skipped_samples:
        logger.warning(
            "Skipped %d samples during multi-pass OPSD batch construction",
            skipped_samples,
        )

    if not teacher_seqs:
        return None

    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])

    return DataProto.from_single_dict(batch_dict), shard_batch_idx, segment_idx


def build_opsd_batch(
    teacher_prompts: list[str],
    student_prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32768,
    teacher_reinject_snippets: Optional[list[Optional[list[int]]]] = None,
    teacher_reinject_positions: Optional[list[Optional[set[int]]]] = None,
) -> Optional[DataProto]:
    """Build paired teacher/student tokenized sequences for OPSD JSD training.

    For each sample, creates two sequences with the same response tokens but
    different prompts:
      - Teacher: sd_prompt (question + teacher solution) + student response
      - Student: sft_prompt (question only) + student response

    The loss_mask marks response positions where JSD should be computed.

    When ``teacher_reinject_snippets`` and ``teacher_reinject_positions`` are
    provided (one entry per sample, may be None/empty to disable per-sample),
    the teacher sequence interleaves the given snippet token ids at the given
    student response positions. The teacher ``loss_mask`` is 0 at snippet
    tokens, so ``_forward_logits_*`` naturally yields teacher logits for
    the student's response tokens in order — aligned 1:1 with the student
    side without a separate index map.

    Args:
        teacher_prompts: JSON-string chat messages with teacher solution (sd_prompt).
        student_prompts: JSON-string chat messages with question only (sft_prompt).
        responses: Student-generated response strings.
        tokenizer: HuggingFace tokenizer.
        max_length: Max total sequence length.
        teacher_reinject_snippets: Per-sample snippet token ids to interleave
            into the teacher response. ``None`` per sample disables reinjection
            for that sample.
        teacher_reinject_positions: Per-sample student response positions at
            which to insert the snippet. ``None``/empty disables.

    Returns:
        DataProto with keys: teacher_input_ids, teacher_attention_mask,
        teacher_position_ids, teacher_loss_mask, student_input_ids,
        student_attention_mask, student_position_ids, student_loss_mask.
        Returns None if no valid samples.
    """
    if not teacher_prompts:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    n = len(teacher_prompts)
    if teacher_reinject_snippets is None:
        teacher_reinject_snippets = [None] * n
    if teacher_reinject_positions is None:
        teacher_reinject_positions = [None] * n

    teacher_seqs = []
    student_seqs = []
    skipped = 0

    for t_prompt, s_prompt, response_text, snippet_ids, positions in zip(
        teacher_prompts, student_prompts, responses,
        teacher_reinject_snippets, teacher_reinject_positions,
    ):
        t_seq = _tokenize_sequence(
            t_prompt, response_text, tokenizer, max_length, pad_token_id,
            reinject_snippet_ids=snippet_ids,
            reinject_positions=positions,
        )
        s_seq = _tokenize_sequence(s_prompt, response_text, tokenizer, max_length, pad_token_id)

        if t_seq is None or s_seq is None:
            skipped += 1
            continue

        teacher_seqs.append(t_seq)
        student_seqs.append(s_seq)

    if skipped:
        logger.warning("Skipped %d samples during OPSD batch construction (too short)", skipped)

    if not teacher_seqs:
        return None

    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])

    return DataProto.from_single_dict(batch_dict)
