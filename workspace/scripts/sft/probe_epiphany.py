#!/usr/bin/env python3
"""Probe an OPSD training run by parsing Ray worker logs.

Usage:
    # Latest run
    python workspace/scripts/sft/probe_epiphany.py

    # Specific run by timestamp (matches outputs/<date>/<time>/)
    python workspace/scripts/sft/probe_epiphany.py 2026-03-25 11-32-36
"""

import glob
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

def main():
    date, time = None, None
    if len(sys.argv) >= 3:
        date, time = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        date = sys.argv[1]

    path = find_log(date, time)
    if not path:
        print("No training run found.")
        sys.exit(1)

    session = re.search(r"session_([^/]+)", path)
    label = session.group(1) if session else "?"
    if "latest" in label:
        resolved = str(Path(path).resolve())
        m = re.search(r"session_([\d][\d-]+_[\d-]+)", resolved)
        if m:
            label = m.group(1)
    print(f"session: {label}\n")

    with open(path) as f:
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
        H = f"{'step':>5}  {'time':>6}  {'loss':>8}  {'gnorm':>6}  {'acc':>6}  {'avg_tok':>7}  {'min_tok':>7}  {'clip':>6}"
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
            print(f"{step:5d}  {time_s:5.0f}s  {loss:8.5f}  {gnorm:6.3f}  {pct(acc)}  {avg_tok:7.0f}  {min_tok:7d}  {pct(clip)}")

    # --- Val ---
    if val_steps:
        benchmarks = []
        for key in val_steps[0]:
            m = re.match(r"val-core/(\w+)/acc/mean@(\d+)", key)
            if m:
                benchmarks.append((m.group(1), int(m.group(2))))

        if benchmarks:
            # Each benchmark gets: acc  maj  tok  (6+2+6+2+6=22 per benchmark block)
            bw = 22  # width per benchmark block
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

if __name__ == "__main__":
    main()
