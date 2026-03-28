#!/usr/bin/env python3
"""Probe an OPSD training run by parsing Ray worker logs.

Usage:
    # Summary of latest run
    python workspace/scripts/sft/probe_epiphany.py

    # Summary of a specific run
    python workspace/scripts/sft/probe_epiphany.py 2026-03-25 11-32-36

    # Epiphany sample from latest run (random step, 1 random sample)
    python workspace/scripts/sft/probe_epiphany.py --samples

    # Epiphany sample from a specific run
    python workspace/scripts/sft/probe_epiphany.py 2026-03-25 11-32-36 --samples

    # Specific run, specific step, more samples
    python workspace/scripts/sft/probe_epiphany.py 2026-03-25 11-32-36 --samples --step 50 --n 10
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path


def find_log(date=None, time=None):
    """Find the OPSDTaskRunner worker log from a Ray session."""
    if date and time:
        pattern = f"/tmp/ray/session_{date}_{time}*/logs/worker*.out"
    else:
        pattern = "/tmp/ray/session_*/logs/worker*.out"
    candidates = sorted(glob.glob(pattern))
    for path in reversed(candidates):
        with open(path) as f:
            head = f.read(512)
        if "OPSDTaskRunner" in head:
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


def session_to_date_time(label):
    """Parse 'yyyy-mm-dd_HH-MM-SS...' into (date, time) for output dir lookup."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", label)
    if m:
        return m.group(1), m.group(2)
    return None, None


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


def find_epiphany_log_dir(date=None, time=None, log_path=None):
    """Resolve the epiphany log directory.

    Logs are co-located in the Hydra timestamp dir:
      outputs/{date}/{time}/detailed_logs/epiphany/

    We try multiple strategies:
      1. Explicit date/time args
      2. Derive date/time from the Ray session label
      3. Fallback: most-recently-modified dir under outputs/
    """
    # Strategy 1: explicit date/time
    if date and time:
        d = os.path.join("outputs", date, time, "detailed_logs", "epiphany")
        if os.path.isdir(d):
            return d

    # Strategy 2: derive from Ray session label
    if log_path:
        label = session_label(log_path)
        s_date, s_time = session_to_date_time(label)
        if s_date and s_time:
            d = os.path.join("outputs", s_date, s_time, "detailed_logs", "epiphany")
            if os.path.isdir(d):
                return d

    # Strategy 3: search all timestamp dirs for epiphany logs (most recent first)
    outputs = Path("outputs")
    if outputs.is_dir():
        candidates = []
        for date_dir in outputs.iterdir():
            if not date_dir.is_dir() or not re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
                continue
            for time_dir in date_dir.iterdir():
                epi = time_dir / "detailed_logs" / "epiphany"
                if epi.is_dir() and any(epi.glob("step_*.json")):
                    candidates.append(epi)
        if candidates:
            # Sort by modification time, newest first
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])

    # Strategy 4: legacy experiment-name dirs (backward compat)
    if outputs.is_dir():
        for entry in sorted(outputs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            d = entry / "detailed_logs" / "epiphany"
            if d.is_dir() and any(d.glob("step_*.json")):
                return str(d)

    return None


def show_summary(log_path):
    """Show the default train/val summary table."""
    label = session_label(log_path)
    print(f"session: {label}\n")

    with open(log_path) as f:
        lines = [l.strip() for l in f if l.startswith("step:")]

    train_steps = []
    val_steps = []
    for line in lines:
        d = parse_kv(line)
        if "opsd/loss" in d:
            train_steps.append(d)
        if "val-core" in line:
            val_steps.append(d)

    if not train_steps and not val_steps:
        print("No steps logged yet.")
        sys.exit(0)

    # --- Train ---
    if train_steps:
        has_epiphany = any("epiphany/avg_tokens" in s for s in train_steps)

        H = f"{'step':>5}  {'time':>6}  {'loss':>8}  {'gnorm':>6}  {'acc':>6}  {'avg_tok':>7}  {'min_tok':>7}  {'clip':>6}"
        if has_epiphany:
            H += f"  {'epi_tok':>7}  {'epi_clip':>8}"
        print("TRAIN")
        print(H)
        print("-" * len(H))
        for s in train_steps:
            step = int(s.get("step", 0))
            time_s = s.get("timing/step_s", 0)
            loss = s.get("opsd/loss", 0)
            gnorm = s.get("opsd/grad_norm", 0)
            n_correct = int(s.get("sd/n_correct", 0))
            batch = int(s.get("sd/batch_size", 1))
            acc = n_correct / batch if batch else 0
            avg_tok = s.get("sd/avg_student_token_count", 0)
            min_tok = int(s.get("sd/min_student_token_count", 0))
            clip = s.get("sd/truncation_rate", 0)
            row = f"{step:5d}  {time_s:5.0f}s  {loss:8.5f}  {gnorm:6.3f}  {pct(acc)}  {avg_tok:7.0f}  {min_tok:7d}  {pct(clip)}"
            if has_epiphany:
                epi_tok = s.get("epiphany/avg_tokens", 0)
                epi_clip = s.get("epiphany/clip_pct", 0)
                row += f"  {epi_tok:7.0f}  {pct(epi_clip, 8)}"
            print(row)

    # --- Val ---
    if val_steps:
        benchmarks = []
        for key in val_steps[0]:
            m = re.match(r"val-core/(\w+)/acc/mean@(\d+)", key)
            if m:
                benchmarks.append((m.group(1), int(m.group(2))))

        if benchmarks:
            H = f"{'step':>5}"
            H2 = f"{'':>5}"
            for bm, n in benchmarks:
                block = f"{'acc@'+str(n):>6}  {'maj@'+str(n):>6}  {'tok':>6}"
                H += f"  {block}"
                lbl = f"{bm:^{len(block)}}"
                H2 += f"  {lbl}"

            print(f"\nVAL")
            print(H2)
            print(H)
            print("-" * len(H))

            for d in val_steps:
                step = int(d.get("step", 0))
                row = f"{step:5d}"
                for bm, n in benchmarks:
                    acc = d.get(f"val-core/{bm}/acc/mean@{n}", 0)
                    maj = d.get(f"val-core/{bm}/acc/maj@{n}/mean", 0)
                    tok = d.get(f"val/{bm}/avg_response_tokens", 0)
                    row += f"  {pct(acc)}  {pct(maj)}  {tok:6.0f}"
                print(row)

    # --- Summary ---
    if train_steps:
        total_time = sum(s.get("timing/step_s", 0) for s in train_steps)
        print(f"\n{len(train_steps)} steps, {total_time/60:.0f}m elapsed, "
              f"~{total_time/len(train_steps):.0f}s/step")


def show_samples(epiphany_dir, step=None, n=1):
    """Show epiphany pipeline samples from JSON logs."""
    step_files = sorted(Path(epiphany_dir).glob("step_*.json"))
    if not step_files:
        print(f"No epiphany log files found in {epiphany_dir}")
        sys.exit(1)

    if step is not None:
        target = f"step_{step:06d}.json"
        matches = [f for f in step_files if f.name == target]
        if not matches:
            available = [int(f.stem.split("_")[1]) for f in step_files]
            print(f"Step {step} not found. Available: {available}")
            sys.exit(1)
        step_files = matches
    else:
        step_files = [random.choice(step_files)]

    for sf in step_files:
        with open(sf) as f:
            data = json.load(f)

        step_num = data.get("step", "?")
        samples = data.get("samples", [])

        samples = random.sample(samples, min(n, len(samples)))

        if not samples:
            print(f"No samples at step {step_num}")
            continue

        for s in samples:
            idx = s.get("sample_idx", "?")
            status = "CORRECT" if s.get("is_correct") else "INCORRECT"
            pred = s.get("prediction", "?")
            gt = s.get("ground_truth", "?")
            t1_tok = s.get("turn1_tokens", "?")
            t2_tok = s.get("turn2_tokens", "?")

            rescued = s.get("rescued", False)
            w = 60
            print("=" * w)
            rescued_tag = " | RESCUED" if rescued else ""
            print(f" Step {step_num} | Sample {idx} | {status} (pred={pred}, gt={gt}){rescued_tag}")
            print("=" * w)

            # Turn 1 Input
            t1_input = s.get("turn1_input", "")
            if isinstance(t1_input, list):
                t1_input = t1_input[0].get("content", "") if t1_input else ""
            print(f"\n---- Turn 1 Input {'-' * (w - 18)}")
            print(t1_input)

            # Turn 1 Output
            t1_out = s.get("turn1_output", "")
            print(f"\n---- Turn 1 Output ({t1_tok} tok) {'-' * max(0, w - 25 - len(str(t1_tok)))}")
            print(t1_out)

            # Turn 2 Input (show only the last user message)
            t2_input = s.get("turn2_input", [])
            t2_user = t2_input[-1].get("content", "") if t2_input else ""
            print(f"\n---- Turn 2 Input {'-' * (w - 18)}")
            print(t2_user)

            # Turn 2 Output Raw
            t2_raw = s.get("turn2_output_raw", "")
            print(f"\n---- Turn 2 Output Raw ({t2_tok} tok) {'-' * max(0, w - 29 - len(str(t2_tok)))}")
            print(t2_raw)

            # Epiphany (stripped)
            epi = s.get("turn2_epiphany", "")
            print(f"\n---- Epiphany (stripped) {'-' * (w - 24)}")
            print(epi)

            # Teacher Input
            teacher = s.get("teacher_input", [])
            teacher_content = teacher[0].get("content", "") if teacher else ""
            print(f"\n---- Teacher Input {'-' * (w - 19)}")
            print(teacher_content)

            print("=" * w)
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Probe an OPSD training run",
        usage="%(prog)s [date] [time] [--samples] [options]",
    )
    parser.add_argument("date", nargs="?", default=None, help="Session date (yyyy-mm-dd)")
    parser.add_argument("time", nargs="?", default=None, help="Session time (hh-mm-ss)")
    parser.add_argument("--samples", action="store_true", help="Show epiphany pipeline samples")
    parser.add_argument("--step", type=int, default=None, help="Specific training step for --samples")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to show (default: 1)")
    args = parser.parse_args()

    if args.samples:
        log_path = find_log(args.date, args.time)
        epiphany_dir = find_epiphany_log_dir(
            date=args.date, time=args.time, log_path=log_path,
        )
        if not epiphany_dir:
            print("Could not find epiphany log directory.")
            print("Looked in outputs/{date}/{time}/detailed_logs/epiphany/")
            sys.exit(1)
        print(f"logs: {epiphany_dir}\n")
        show_samples(epiphany_dir, step=args.step, n=args.n)
    else:
        path = find_log(args.date, args.time)
        if not path:
            print("No training run found.")
            sys.exit(1)
        show_summary(path)


if __name__ == "__main__":
    main()
