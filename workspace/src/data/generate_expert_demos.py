"""
Generate expert demonstrations for the first 3200 training questions.

Calls an OpenAI model to produce step-by-step solutions using the exact same
prompt the student model sees. Verifies correctness via an LLM judge and retries
on failure. If all retries fail, makes a final GT-assisted attempt where the
correct answer is provided in the prompt.

Uses async concurrency to parallelize across prompts.

Output:
  workspace/data/expert_demonstrations/expert_demos_3200.parquet
  workspace/data/expert_demonstrations/expert_demos_samples.json  (first 20)

Usage:
    # Test with 10 questions
    python workspace/src/data/generate_expert_demos.py --limit 10

    # Full run (all 3200)
    python workspace/src/data/generate_expert_demos.py

    # Custom model
    python workspace/src/data/generate_expert_demos.py --model gpt-5.4-2026-03-05

    # Custom max retries
    python workspace/src/data/generate_expert_demos.py --max-retries 3

    # Retry only empty entries
    python workspace/src/data/generate_expert_demos.py --retry-empty

    # Retry empty entries with more GT-assisted attempts
    python workspace/src/data/generate_expert_demos.py --retry-empty --gt-retries 3
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_INPUT_PARQUET = _WORKSPACE_ROOT / "data" / "length_prune_concise" / "self_distill_prompts.parquet"
_OUTPUT_DIR = _WORKSPACE_ROOT / "data" / "expert_demonstrations"

DEFAULT_MODEL = "gpt-5.4-2026-03-05"
JUDGE_MODEL = "gpt-5.4-nano-2026-03-17"
N_TRAIN = 3200
MAX_RETRIES = 1
GT_RETRIES = 1
CONCURRENCY = 32
BATCH_SIZE = 320
N_INSPECT_SAMPLES = 20

JUDGE_SYSTEM = (
    "You are a math answer checker. You will be given a ground truth answer and "
    "a candidate answer extracted from a solution. Determine if they are "
    "mathematically equivalent.\n\n"
    "Respond with exactly one word: YES or NO."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_answer(response_text: str) -> str:
    """Extract the answer after 'Answer:' from the response."""
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line_stripped = line.strip()
        if line_stripped.lower().startswith("answer:"):
            return line_stripped[len("answer:"):].strip().strip("$").strip()
    return ""


async def generate_solution(
    client: AsyncOpenAI, model: str, user_content: str, sem: asyncio.Semaphore
) -> tuple[str, int]:
    """Generate a single expert solution using the exact student prompt."""
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            max_completion_tokens=2048,
        )
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return resp.choices[0].message.content or "", tokens


async def generate_gt_assisted(
    client: AsyncOpenAI, model: str, user_content: str, ground_truth: str,
    sem: asyncio.Semaphore,
) -> tuple[str, int]:
    """Generate a solution with the correct answer provided in the prompt."""
    assisted_content = (
        f"{user_content}\n\n"
        f"The correct answer is: {ground_truth}\n"
        f"Solve the problem and explain why this is the answer."
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": assisted_content}],
            max_completion_tokens=2048,
        )
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return resp.choices[0].message.content or "", tokens


async def judge_correctness(
    client: AsyncOpenAI, ground_truth: str, candidate_answer: str,
    sem: asyncio.Semaphore,
) -> bool:
    """Use LLM judge to check if candidate matches ground truth."""
    if not candidate_answer:
        return False

    prompt = (
        f"Ground truth answer: {ground_truth}\n"
        f"Candidate answer: {candidate_answer}\n\n"
        f"Are these mathematically equivalent? Reply YES or NO."
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=16,
        )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    idx: int,
    user_content: str,
    ground_truth: str,
    max_retries: int,
    gt_retries: int,
    sem: asyncio.Semaphore,
    counter: dict,
    total: int,
    t0: float,
) -> dict:
    """Generate expert demo for one question with retries.

    Retries cover both wrong answers and transient API errors.

    Returns a result dict.
    """
    for attempt in range(1, max_retries + 1):
        try:
            solution, tok = await generate_solution(client, model, user_content, sem)
            answer = extract_answer(solution)
            correct = await judge_correctness(client, ground_truth, answer, sem)
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            else:
                # Fall through to GT-assisted attempt
                break
        if correct:
            counter["done"] += 1
            counter["correct"] += 1
            _print_progress(counter, total, t0, idx, "OK", attempt, answer, ground_truth, tok)
            return {
                "question_idx": idx,
                "expert_demonstration": solution,
                "extracted_answer": answer,
                "is_correct": True,
                "attempts": attempt,
                "gt_assisted": False,
            }

    # All retries exhausted — GT-assisted attempts
    for gt_attempt in range(1, gt_retries + 1):
        try:
            solution, tok = await generate_gt_assisted(client, model, user_content, ground_truth, sem)
        except Exception as e:
            if gt_attempt < gt_retries:
                await asyncio.sleep(min(2 ** gt_attempt, 30))
                continue
            counter["done"] += 1
            _print_progress(counter, total, t0, idx, f"FAILED({e})", max_retries + gt_attempt, "", ground_truth, 0)
            return {
                "question_idx": idx,
                "expert_demonstration": "",
                "extracted_answer": "",
                "is_correct": False,
                "attempts": max_retries + gt_attempt,
                "gt_assisted": True,
            }
        answer = extract_answer(solution)
        if answer or gt_attempt == gt_retries:
            counter["done"] += 1
            counter["gt_assisted"] += 1
            _print_progress(counter, total, t0, idx, "GT-ASSIST", max_retries + gt_attempt, answer, ground_truth, tok)
            return {
                "question_idx": idx,
                "expert_demonstration": solution,
                "extracted_answer": answer,
                "is_correct": True,
                "attempts": max_retries + gt_attempt,
                "gt_assisted": True,
            }


def _print_progress(counter, total, t0, idx, tag, attempts, answer, gt, tok=0):
    done = counter["done"]
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(
        f"  [{done}/{total}] idx={idx} {tag} (attempts={attempts}, "
        f"tok={tok}, pred={answer}, gt={gt}) "
        f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args):
    # Load env
    load_dotenv(_WORKSPACE_ROOT.parent / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # Load training data
    print(f"Loading {_INPUT_PARQUET}")
    df = pd.read_parquet(_INPUT_PARQUET)
    df = df.head(N_TRAIN)  # first 3200 (matches training subset)
    n = args.limit if args.limit else len(df)
    df = df.head(n).reset_index(drop=True)
    print(f"Generating expert demos for {len(df)} questions (model={args.model}, concurrency={args.concurrency})")

    # Parse sft_prompt to get the exact user content for each question
    user_contents = []
    for i in range(len(df)):
        sft_prompt = df.iloc[i]["sft_prompt"]
        msgs = json.loads(sft_prompt) if isinstance(sft_prompt, str) else sft_prompt
        user_contents.append(msgs[0]["content"])

    # Run async generation in batches
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = _OUTPUT_DIR / "expert_demos_3200.parquet"
    samples_path = _OUTPUT_DIR / "expert_demos_samples.json"

    sem = asyncio.Semaphore(args.concurrency)
    counter = {"done": 0, "correct": 0, "gt_assisted": 0}
    t0 = time.time()
    batch_size = BATCH_SIZE

    all_records = {}  # idx -> record dict
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        tasks = []
        for i in range(batch_start, batch_end):
            tasks.append(
                generate_one(
                    client, args.model, i,
                    user_contents[i], df.iloc[i]["ground_truth"],
                    args.max_retries, args.gt_retries, sem, counter, len(df), t0,
                )
            )
        batch_results = await asyncio.gather(*tasks)

        # Convert to records and append
        for r in batch_results:
            idx = r["question_idx"]
            all_records[idx] = {
                "question": df.iloc[idx]["question"],
                "ground_truth": df.iloc[idx]["ground_truth"],
                "expert_demonstration": r["expert_demonstration"],
                "extracted_answer": r["extracted_answer"],
                "is_correct": r["is_correct"],
                "attempts": r["attempts"],
                "gt_assisted": r["gt_assisted"],
            }

        # Save completed records in original order
        records_so_far = [all_records[i] for i in sorted(all_records.keys())]

        out_df = pd.DataFrame(records_so_far)
        out_df.to_parquet(parquet_path)

        # Inspection samples (first 20, full content)
        n_samples = min(N_INSPECT_SAMPLES, len(records_so_far))
        samples = []
        for i in range(n_samples):
            rec = records_so_far[i]
            samples.append({
                "index": i,
                "question": rec["question"],
                "ground_truth": rec["ground_truth"],
                "extracted_answer": rec["extracted_answer"],
                "is_correct": rec["is_correct"],
                "attempts": rec["attempts"],
                "gt_assisted": rec["gt_assisted"],
                "expert_demonstration": rec["expert_demonstration"],
            })
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    # Summary
    elapsed = time.time() - t0
    n_correct = counter["correct"]
    n_gt = counter["gt_assisted"]
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  {n_correct}/{len(df)} correct on own ({n_correct/len(df)*100:.1f}%)")
    print(f"  {n_gt} required GT-assisted generation")
    print(f"  Parquet: {parquet_path}")
    print(f"  Samples: {samples_path}")


async def async_retry_empty(args):
    """Reload existing parquet, regenerate only entries with empty expert_demonstration."""
    load_dotenv(_WORKSPACE_ROOT.parent / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    parquet_path = _OUTPUT_DIR / "expert_demos_3200.parquet"

    existing = pd.read_parquet(parquet_path)
    empty_mask = existing["expert_demonstration"] == ""
    empty_indices = list(existing[empty_mask].index)
    print(f"Loaded {len(existing)} entries, {len(empty_indices)} empty — regenerating those")

    if not empty_indices:
        print("Nothing to retry.")
        return

    # Parse sft_prompt for the empty entries
    train = pd.read_parquet(_INPUT_PARQUET).head(N_TRAIN)
    user_contents = {}
    for i in empty_indices:
        sft_prompt = train.iloc[i]["sft_prompt"]
        msgs = json.loads(sft_prompt) if isinstance(sft_prompt, str) else sft_prompt
        user_contents[i] = msgs[0]["content"]

    sem = asyncio.Semaphore(args.concurrency)
    counter = {"done": 0, "correct": 0, "gt_assisted": 0}
    t0 = time.time()
    total = len(empty_indices)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_indices = empty_indices[batch_start:batch_start + BATCH_SIZE]
        tasks = []
        for i in batch_indices:
            tasks.append(
                generate_one(
                    client, args.model, i,
                    user_contents[i], existing.iloc[i]["ground_truth"],
                    args.max_retries, args.gt_retries, sem, counter, total, t0,
                )
            )
        batch_results = await asyncio.gather(*tasks)

        # Update existing dataframe
        for r in batch_results:
            idx = r["question_idx"]
            existing.at[idx, "expert_demonstration"] = r["expert_demonstration"]
            existing.at[idx, "extracted_answer"] = r["extracted_answer"]
            existing.at[idx, "is_correct"] = r["is_correct"]
            existing.at[idx, "attempts"] = r["attempts"]
            existing.at[idx, "gt_assisted"] = r["gt_assisted"]

        # Save after each batch
        existing.to_parquet(parquet_path)

    still_empty = (existing["expert_demonstration"] == "").sum()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Retried: {total}, still empty: {still_empty}")
    print(f"  Parquet: {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate expert demonstrations")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions (for testing). Default: all 3200.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=MAX_RETRIES,
        help=f"Max generation attempts per question (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--gt-retries", type=int, default=GT_RETRIES,
        help=f"Max GT-assisted attempts after retries exhausted (default: {GT_RETRIES})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
        help=f"Max concurrent API calls (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--retry-empty", action="store_true",
        help="Load existing parquet and only regenerate entries with empty expert_demonstration.",
    )
    args = parser.parse_args()
    if args.retry_empty:
        asyncio.run(async_retry_empty(args))
    else:
        asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
