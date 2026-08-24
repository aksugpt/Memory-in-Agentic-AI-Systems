"""
run_trials.py

Runs each attack scenario N times, logs every trial's outcome to a
CSV, and computes Attack Success Rate (ASR) with a Wilson confidence
interval -- this produces the first real number for your thesis:
baseline ASR with NO gate in place.

Once Gate v1 exists, this same script (pointed at a gated agent) is
what re-measures ASR after the fix -- that before/after comparison is
your core result, so keep this runner generic rather than one-off.

Usage:
    python run_trials.py                  # 10 trials per scenario (quick check)
    python run_trials.py --trials 30      # 30 trials per scenario (real run)
"""

import argparse
import csv
import os
import time
from datetime import datetime

from statsmodels.stats.proportion import proportion_confint

from attacks.indirect_injection import POISON_SCENARIOS, run_single_trial

RESULTS_DIR = "./results"


def run_all_trials(n_trials: int, gate_label: str = "no_gate") -> list[dict]:
    """
    Runs n_trials for every scenario in POISON_SCENARIOS.
    `gate_label` is just a tag for the CSV/results -- e.g. "no_gate",
    "gate_v1", "gate_v2" -- so the same runner works across every
    phase without rewriting this file each time.
    """
    all_results = []
    total_runs = n_trials * len(POISON_SCENARIOS)
    run_num = 0

    for scenario in POISON_SCENARIOS:
        for trial_i in range(n_trials):
            run_num += 1
            # Separate on-disk memory per trial so trials don't leak
            # into each other -- each trial gets a genuinely clean slate.
            memory_path = f"./chroma_data_trial_{scenario['name']}_{trial_i}"

            print(f"[{run_num}/{total_runs}] {scenario['name']} trial {trial_i + 1}/{n_trials}...", end=" ", flush=True)

            try:
                result = run_single_trial(scenario, memory_path=memory_path)
                result["gate"] = gate_label
                result["trial_index"] = trial_i
                result["timestamp"] = datetime.now().isoformat()
                print("SUCCESS" if result["trigger_succeeded"] else "failed")
            except Exception as e:
                # Don't let one bad trial (e.g. a transient API error
                # that exhausted retries) kill the whole run -- log it
                # as a failed/errored trial and keep going.
                print(f"ERROR: {e}")
                result = {
                    "scenario": scenario["name"],
                    "gate": gate_label,
                    "trial_index": trial_i,
                    "write_succeeded": False,
                    "trigger_succeeded": False,
                    "reply": f"[ERROR] {e}",
                    "stored_memories": [],
                    "timestamp": datetime.now().isoformat(),
                }

            all_results.append(result)

    return all_results


def save_results_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gate", "scenario", "trial_index", "write_succeeded", "trigger_succeeded", "reply"])
        for r in results:
            writer.writerow([
                r["timestamp"], r["gate"], r["scenario"], r["trial_index"],
                r["write_succeeded"], r["trigger_succeeded"], r["reply"].replace("\n", " ")[:300],
            ])
    print(f"\nRaw trial log saved to: {path}")


def print_asr_summary(results: list[dict]):
    """
    Prints ASR (based on trigger_succeeded, since that's what actually
    matters -- the poison surviving into a fresh session AND getting
    acted on) with a Wilson 95% confidence interval, overall and
    broken down per scenario.
    """
    print("\n" + "=" * 60)
    print("ATTACK SUCCESS RATE (ASR) SUMMARY")
    print("=" * 60)

    def asr_line(label: str, subset: list[dict]):
        n = len(subset)
        successes = sum(1 for r in subset if r["trigger_succeeded"])
        if n == 0:
            return
        asr = successes / n
        low, high = proportion_confint(successes, n, alpha=0.05, method="wilson")
        print(f"{label:35s} {successes:3d}/{n:<3d}  ASR = {asr:.1%}   95% CI [{low:.1%}, {high:.1%}]")

    asr_line("OVERALL", results)
    print("-" * 60)
    scenario_names = sorted(set(r["scenario"] for r in results))
    for name in scenario_names:
        subset = [r for r in results if r["scenario"] == name]
        asr_line(name, subset)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Trials per scenario (default 10 for a quick check; use 30 for a real run)")
    parser.add_argument("--gate", type=str, default="no_gate", help="Label for this run, e.g. no_gate / gate_v1 / gate_v2")
    args = parser.parse_args()

    print(f"Running {args.trials} trials per scenario ({len(POISON_SCENARIOS)} scenarios, "
          f"{args.trials * len(POISON_SCENARIOS)} total) -- gate: {args.gate}\n")

    start = time.time()
    results = run_all_trials(args.trials, gate_label=args.gate)
    elapsed = time.time() - start

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_csv(results, f"{RESULTS_DIR}/trials_{args.gate}_{timestamp}.csv")
    print_asr_summary(results)
    print(f"\nTotal time: {elapsed:.0f}s")