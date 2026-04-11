#!/usr/bin/env python3
"""Probe a GRPO training run by parsing Ray worker logs.

Usage:
    # Summary of latest run
    python workspace/scripts/rl/probe_grpo.py

    # Summary of a specific run
    python workspace/scripts/rl/probe_grpo.py 2026-04-03 10-15-00

    # With per-benchmark token counts (requires tokenizer)
    python workspace/scripts/rl/probe_grpo.py --model Qwen/Qwen3-8B
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def find_log(date=None, time=None):
    """Find the TaskRunner worker log from a Ray session."""
    if date and time:
        pattern = f"/tmp/ray/session_{date}_{time}*/logs/worker*.out"
    else:
        pattern = "/tmp/ray/session_*/logs/worker*.out"
    candidates = sorted(glob.glob(pattern))
    for path in reversed(candidates):
        with open(path) as f:
            head = f.read(512)
        if "TaskRunner" in head:
            return path
    return None


def session_label(log_path):
    """Extract human-readable session label from log path."""
    session = re.search(r"session_([^/]+)", log_path)
    label = session.group(1) if session else "?"
    if "latest" in label:
        resolved = str(Path(log_path).resolve())
        m = re.search(r"session_([\d][\d-]+_[\d-]+)", resolved)
        if m:
            label = m.group(1)
    return label


def parse_kv(line):
    d = {}
    for part in line.split(" - "):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def pct(v, w=6):
    return f"{v * 100:{w - 1}.1f}%"


# =============================================================================
# Val generation JSONL parsing for per-benchmark token counts
# =============================================================================

# Val datasets are loaded in this order; each prompt repeated n times (default 8)
VAL_BENCHMARKS = [
    ("math", 500),
    ("aime24", 30),
    ("aime25", 30),
]


def find_val_gen_dir(log_path=None):
    """Find the val_generations directory for the current run."""
    outputs = Path("outputs")
    if not outputs.is_dir():
        return None

    candidates = []
    for entry in outputs.iterdir():
        vg = entry / "val_generations"
        if vg.is_dir() and any(vg.glob("*.jsonl")):
            candidates.append(vg)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def compute_val_token_stats(val_gen_dir, model_path=None):
    """Parse val generation JSONL files and compute per-benchmark avg token counts.

    Returns: {step: {benchmark_name: avg_tokens}}
    """
    tokenizer = None
    if model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: could not load tokenizer from {model_path}: {e}",
                  file=sys.stderr)

    jsonl_files = sorted(Path(val_gen_dir).glob("*.jsonl"))
    if not jsonl_files:
        return {}

    result = {}
    for jf in jsonl_files:
        step = int(jf.stem)
        with open(jf) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        if not entries:
            continue

        # Figure out n (responses per prompt) from total count and benchmark sizes
        total_prompts = sum(size for _, size in VAL_BENCHMARKS)
        if len(entries) % total_prompts == 0:
            n = len(entries) // total_prompts
        else:
            n = 8  # fallback

        # Split entries by benchmark
        offset = 0
        step_stats = {}
        for bm_name, bm_size in VAL_BENCHMARKS:
            count = bm_size * n
            bm_entries = entries[offset:offset + count]
            offset += count

            if not bm_entries:
                continue

            if tokenizer:
                lengths = [len(tokenizer.encode(e["output"])) for e in bm_entries]
            else:
                # Rough estimate: ~3.5 chars per token for English/math
                lengths = [len(e["output"]) / 3.5 for e in bm_entries]

            step_stats[bm_name] = sum(lengths) / len(lengths)

        result[step] = step_stats

    return result


def show_summary(log_path, model_path=None):
    """Show the default train/val summary table."""
    label = session_label(log_path)
    print(f"session: {label}\n")

    with open(log_path) as f:
        lines = [l.strip() for l in f if l.startswith("step:")]

    train_steps = []
    val_steps = []
    for line in lines:
        d = parse_kv(line)
        if "actor/policy_loss" in d or "actor/loss" in d or "actor/pg_loss" in d:
            train_steps.append(d)
        if "val-core" in line:
            val_steps.append(d)

    if not train_steps and not val_steps:
        print("No steps logged yet.")
        sys.exit(0)

    # --- Train ---
    if train_steps:
        H = (f"{'step':>5}  {'time':>6}  {'loss':>8}  {'gnorm':>6}"
             f"  {'reward':>7}  {'avg_tok':>7}  {'clip':>6}"
             f"  {'mixed':>6}  {'pass':>6}")
        print("TRAIN")
        print(H)
        print("-" * len(H))
        for s in train_steps:
            step = int(s.get("step", s.get("training/global_step", 0)))
            time_s = s.get("timing_s/step", s.get("timing/step", s.get("timing/step_s", 0)))
            loss = s.get("actor/pg_loss", s.get("actor/policy_loss", s.get("actor/loss", 0)))
            gnorm = s.get("actor/grad_norm", 0)
            reward = s.get("critic/rewards/mean", s.get("reward/mean", 0))
            avg_tok = s.get("response_length/mean", 0)
            clip = s.get("response_length/clip_ratio", 0)
            frac_mixed = s.get("grpo/frac_mixed", -1)
            pass_rate = s.get("grpo/mean_group_pass_rate", -1)
            mixed_s = pct(frac_mixed) if frac_mixed >= 0 else "   ---"
            pass_s = pct(pass_rate) if pass_rate >= 0 else "   ---"
            print(f"{step:5d}  {time_s:5.0f}s  {loss:8.4f}  {gnorm:6.3f}"
                  f"  {reward:7.4f}  {avg_tok:7.0f}  {pct(clip)}"
                  f"  {mixed_s}  {pass_s}")

    # --- Val ---
    if val_steps:
        # Load per-benchmark token stats from JSONL dumps
        val_gen_dir = find_val_gen_dir(log_path)
        val_tok_stats = {}
        if val_gen_dir:
            val_tok_stats = compute_val_token_stats(val_gen_dir, model_path)

        benchmarks = []
        for key in val_steps[0]:
            m = re.match(r"val-core/(\w+)/acc/mean@(\d+)", key)
            if m:
                benchmarks.append((m.group(1), int(m.group(2))))

        if benchmarks:
            has_tok = bool(val_tok_stats)

            H = f"{'step':>5}"
            H2 = f"{'':>5}"
            for bm, n in benchmarks:
                block = f"{'acc@'+str(n):>6}  {'maj@'+str(n):>6}"
                if has_tok:
                    block += f"  {'tok':>6}"
                H += f"  {block}"
                lbl = f"{bm:^{len(block)}}"
                H2 += f"  {lbl}"

            print(f"\nVAL")
            print(H2)
            print(H)
            print("-" * len(H))

            first_train_step = int(train_steps[0].get("step", train_steps[0].get("training/global_step", 0))) if train_steps else None
            for i, d in enumerate(val_steps):
                step = int(d.get("step", 0))
                if i == 0 and first_train_step is not None and step == first_train_step:
                    step = 0
                row = f"{step:5d}"
                step_tok = val_tok_stats.get(step, {})
                for bm, n in benchmarks:
                    acc = d.get(f"val-core/{bm}/acc/mean@{n}", 0)
                    maj = d.get(f"val-core/{bm}/acc/maj@{n}/mean", 0)
                    row += f"  {pct(acc)}  {pct(maj)}"
                    if has_tok:
                        tok = step_tok.get(bm, 0)
                        row += f"  {tok:6.0f}" if tok else f"  {'---':>6}"
                print(row)

    # --- Summary ---
    if train_steps:
        total_time = sum(
            s.get("timing_s/step", s.get("timing/step", s.get("timing/step_s", 0)))
            for s in train_steps)
        print(f"\n{len(train_steps)} steps, {total_time/60:.0f}m elapsed, "
              f"~{total_time/len(train_steps):.0f}s/step")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a GRPO training run",
        usage="%(prog)s [date] [time] [--model MODEL]",
    )
    parser.add_argument("date", nargs="?", default=None,
                        help="Session date (yyyy-mm-dd)")
    parser.add_argument("time", nargs="?", default=None,
                        help="Session time (hh-mm-ss)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for tokenizer (e.g. Qwen/Qwen3-8B)")
    args = parser.parse_args()

    path = find_log(args.date, args.time)
    if not path:
        print("No GRPO training run found.")
        sys.exit(1)
    show_summary(path, model_path=args.model)


if __name__ == "__main__":
    main()
