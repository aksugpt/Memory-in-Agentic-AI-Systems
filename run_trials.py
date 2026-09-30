"""
run_trials.py

Runs each attack scenario N times, logs every trial's outcome to a
CSV, and computes Attack Success Rate (ASR) with a Wilson confidence
interval -- this produces the first real number for your thesis:
baseline ASR with NO gate in place.

Once Gate v1 exists, this same script (pointed at a gated agent) is
what re-measures ASR after the fix -- that before/after comparison is
your core result, so keep this runner generic rather than one-off.

Each trial's row is written to the CSV as soon as it finishes, so an
interrupted run keeps everything done so far and can be continued
with --resume.

Usage:
    python run_trials.py                  # 10 trials per scenario (quick check)
    python run_trials.py --trials 30      # 30 trials per scenario (real run)
    python run_trials.py --trials 30 --gate gate_v2 --delivery both --resume results/<that run>.csv
    python run_trials.py --summarize results/<some run>.csv
"""

import argparse
import csv
import os
import shutil
import time
from datetime import datetime

from statsmodels.stats.proportion import proportion_confint

from attacks.indirect_injection import SUITES, DELIVERY_MODES, run_single_trial

RESULTS_DIR = "./results"

class DailyLimitReached(Exception):
    """Groq's per-day token/request quota is exhausted -- retrying won't help until it resets."""


def _is_daily_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg and ("per day" in msg or "(tpd)" in msg or "(rpd)" in msg)


CSV_COLUMNS = ["timestamp", "gate", "scenario", "trial_index", "write_succeeded", "trigger_succeeded",
               "gate_action", "reply", "delivery", "poison_origin", "error", "kind", "action_attempted",
               "auto_executed", "held_for_confirmation"]


def _resolve_gate_version(gate_label: str) -> str | None:
    """
    Maps the CLI/CSV label to the actual gate_version Agent understands.
    "no_gate" -> None (original vulnerable baseline, unchanged).
    "gate_v1" -> "v1". "gate_v2" -> "v2".
    Keeping the label and the mechanism separate means the CSV stays
    human-readable while agent.py only needs to know about version
    strings, not this project's naming conventions.
    """
    if gate_label == "no_gate":
        return None
    if gate_label.startswith("gate_"):
        return gate_label[len("gate_"):]  # "gate_v1" -> "v1"
    raise ValueError(f"Unrecognized --gate value: {gate_label!r} (expected no_gate / gate_v1 / gate_v2)")


def _csv_row(r: dict) -> list:
    return [
        r["timestamp"], r["gate"], r["scenario"], r["trial_index"],
        r["write_succeeded"], r["trigger_succeeded"], r.get("gate_action", "n/a"),
        r["reply"].replace("\n", " ")[:300],
        r.get("delivery", "upload"), r.get("poison_origin", "n/a"), r.get("error", False),
        r.get("kind", "attack"), r.get("action_attempted"),
        r.get("auto_executed"), r.get("held_for_confirmation"),
    ]


def load_results_csv(path: str) -> list[dict]:
    """
    Read a results CSV back into result dicts (booleans restored).
    Older CSVs without the delivery/poison_origin/error columns load
    with defaults; errored trials in them are recognized by the
    "[ERROR]" reply prefix.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["trial_index"] = int(r["trial_index"])
        r["write_succeeded"] = r["write_succeeded"] == "True"
        r["trigger_succeeded"] = r["trigger_succeeded"] == "True"
        r["delivery"] = r.get("delivery") or "upload"
        r["kind"] = r.get("kind") or "attack"
        for col in ("action_attempted", "auto_executed", "held_for_confirmation"):
            r[col] = {"True": True, "False": False}.get(r.get(col))
        r["error"] = r.get("error") == "True" or r["reply"].startswith("[ERROR]")
    return rows


def run_all_trials(n_trials: int, csv_path: str, gate_label: str = "no_gate", keep_chroma: bool = False,
                   deliveries: tuple[str, ...] = ("upload",), resume: bool = False, suite: str = "text") -> list[dict]:
    """
    Runs n_trials for every scenario in SUITES[suite] (default: the
    original text-only POISON_SCENARIOS).
    `gate_label` is both the CSV tag AND (via _resolve_gate_version)
    what actually toggles the gate on/off in the Agent -- "no_gate"
    truly runs ungated, "gate_v1" truly routes through Gate v1. This
    is what makes the before/after ASR comparison real, not just two
    differently-labeled runs of the same ungated code.

    Each trial's row is appended to csv_path and flushed as soon as
    the trial finishes, so a crash, Ctrl+C or closed window loses at
    most the trial in progress. With resume=True, trials already in
    csv_path are skipped; errored rows are dropped and retried.

    Each trial gets its own on-disk Chroma folder for isolation, then
    it's deleted right after the trial finishes (unless keep_chroma=True)
    -- what matters long-term is the CSV row, not the raw vector store,
    and leaving 90+ folders on disk after a full run is just clutter.

    Returns all rows in the CSV afterwards (earlier + this run).
    """
    gate_version = _resolve_gate_version(gate_label)
    # Benign scenarios don't use a document, so delivery doesn't apply:
    # run them once (under the first delivery mode) rather than per mode.
    jobs = [(d, sc, t) for d in deliveries for sc in SUITES[suite] for t in range(n_trials)
            if sc.get("kind", "attack") == "attack" or d == deliveries[0]]
    total_runs = len(jobs)
    run_num = 0

    previous = []
    if resume and os.path.exists(csv_path):
        previous = [r for r in load_results_csv(csv_path) if not r["error"]]
        mismatched = {r["gate"] for r in previous} - {gate_label}
        if mismatched:
            raise ValueError(f"--resume CSV was run with gate {mismatched}, but this run is --gate {gate_label}")
        print(f"Resuming {csv_path}: {len(previous)} finished trials kept, skipping them.\n")
    done = {(r["scenario"], r["delivery"], r["trial_index"]) for r in previous}
    all_results = list(previous)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # Rewrite the file with just the header + kept rows (drops errored
    # rows from an earlier attempt), then append new trials as they finish.
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for r in previous:
            writer.writerow(_csv_row(r))
        f.flush()

        for delivery, scenario, trial_i in jobs:
            run_num += 1
            if (scenario["name"], delivery, trial_i) in done:
                continue
            # Separate on-disk memory per trial so trials don't leak
            # into each other -- each trial gets a genuinely clean slate.
            memory_path = f"./chroma_data_trial_{scenario['name']}_{delivery}_{trial_i}"

            print(f"[{run_num}/{total_runs}] {scenario['name']} [{delivery}] trial {trial_i + 1}/{n_trials}...", end=" ", flush=True)

            try:
                result = run_single_trial(scenario, memory_path=memory_path, gate_version=gate_version, delivery=delivery)
                result["error"] = False
                status = "SUCCESS" if result["trigger_succeeded"] else "failed"
                if result.get("gate_action") and result["gate_action"] != "n/a":
                    status += f" (gate: {result['gate_action']})"
                print(status)
            except Exception as e:
                if _is_daily_limit(e):
                    # Every remaining trial would fail the same way;
                    # stop now, don't write an error row for this one.
                    print("DAILY LIMIT")
                    raise DailyLimitReached(str(e)) from e
                # Don't let one bad trial (e.g. a transient API error
                # that exhausted retries) kill the whole run -- log it
                # as an errored trial (excluded from ASR) and keep going.
                print(f"ERROR: {e}")
                result = {
                    "scenario": scenario["name"],
                    "delivery": delivery,
                    "poison_origin": "n/a",
                    "kind": scenario.get("kind", "attack"),
                    "action_attempted": None,
                    "auto_executed": None,
                    "held_for_confirmation": None,
                    "write_succeeded": False,
                    "trigger_succeeded": False,
                    "gate_action": "n/a",
                    "reply": f"[ERROR] {e}",
                    "error": True,
                }
            result["gate"] = gate_label
            result["trial_index"] = trial_i
            result["timestamp"] = datetime.now().isoformat()

            writer.writerow(_csv_row(result))
            f.flush()
            all_results.append(result)

            if not keep_chroma and os.path.exists(memory_path):
                shutil.rmtree(memory_path, ignore_errors=True)

    print(f"\nRaw trial log saved to: {csv_path}")
    return all_results


def print_asr_summary(results: list[dict]):
    """
    Prints ASR (based on trigger_succeeded, since that's what actually
    matters -- the poison surviving into a fresh session AND getting
    acted on) with a Wilson 95% confidence interval, overall and
    broken down per delivery mode and scenario.

    Errored trials (API failures after all retries) are EXCLUDED from
    the denominator: the attack never ran, so counting them as
    "attack failed" would make a defended run look better than it is.
    """
    errored = [r for r in results if r.get("error")]
    benign = [r for r in results if not r.get("error") and r.get("kind") == "benign"]
    results = [r for r in results if not r.get("error") and r.get("kind", "attack") == "attack"]

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
    deliveries = sorted(set(r.get("delivery", "upload") for r in results))
    for delivery in deliveries:
        by_delivery = [r for r in results if r.get("delivery", "upload") == delivery]
        if len(deliveries) > 1:
            asr_line(f"[{delivery}] all", by_delivery)
        for name in sorted(set(r["scenario"] for r in by_delivery)):
            asr_line(f"[{delivery}] {name}", [r for r in by_delivery if r["scenario"] == name])

    gate_actions = [r.get("gate_action") for r in results if r.get("gate_action") not in (None, "n/a")]
    if gate_actions:
        print("-" * 60)
        print("Gate action breakdown (what the gate decided on the poisoned write):")
        for action in sorted(set(gate_actions)):
            count = gate_actions.count(action)
            print(f"  {action:12s} {count:3d}/{len(gate_actions)}")

    attempted = [r for r in results if r.get("action_attempted") is not None]
    if attempted:
        print("-" * 60)
        print("Action scenarios -- attacker's address PROPOSED / HELD for confirmation / EXECUTED:")
        for name in sorted(set(r["scenario"] for r in attempted)):
            sub = [r for r in attempted if r["scenario"] == name]
            a = sum(1 for r in sub if r["action_attempted"])
            h = sum(1 for r in sub if r.get("held_for_confirmation"))
            e = sum(1 for r in sub if r["trigger_succeeded"])
            print(f"  {name:33s} proposed {a:3d}/{len(sub):<3d} held {h:3d}  executed {e:3d}")

    if benign:
        print("-" * 60)
        print("LEGIT UTILITY (benign scenarios -- higher is better):")
        for name in sorted(set(r["scenario"] for r in benign)):
            sub = [r for r in benign if r["scenario"] == name]
            ok = sum(1 for r in sub if r["trigger_succeeded"])
            low, high = proportion_confint(ok, len(sub), alpha=0.05, method="wilson")
            print(f"  {name:33s} {ok:3d}/{len(sub):<3d}  utility = {ok / len(sub):.1%}   95% CI [{low:.1%}, {high:.1%}]")
            if any(r.get("held_for_confirmation") is not None for r in sub):
                auto = sum(1 for r in sub if r.get("auto_executed"))
                held = sum(1 for r in sub if r.get("held_for_confirmation"))
                print(f"  {'':33s} auto-executed {auto}/{len(sub)}, held for confirmation {held}/{len(sub)}")

    if errored:
        print("-" * 60)
        print(f"Errored trials excluded from ASR: {len(errored)} (rerun with --resume to retry them)")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Trials per scenario (default 10 for a quick check; use 30 for a real run)")
    parser.add_argument("--gate", type=str, default="no_gate", help="Label for this run, e.g. no_gate / gate_v1 / gate_v2")
    parser.add_argument("--suite", choices=sorted(SUITES), default="text", help="Scenario set: text (original 3 text-marker scenarios, default -- matches earlier runs), action (send_email tool + act gate, incl. a benign utility control), or all")
    parser.add_argument("--delivery", choices=[*DELIVERY_MODES, "both"], default="upload", help="How the poisoned document reaches the agent: upload (process_document, default -- matches earlier runs), paste (inside an ordinary chat message), or both")
    parser.add_argument("--keep-chroma", action="store_true", help="Keep each trial's chroma_data_trial_* folder instead of deleting it after the trial (useful for manually inspecting one trial's memory state)")
    parser.add_argument("--resume", type=str, default=None, metavar="CSV", help="Continue an interrupted run from its results CSV: finished trials are skipped, errored ones retried. Use the same --trials/--gate/--delivery as the original run.")
    parser.add_argument("--summarize", type=str, default=None, metavar="CSV", help="Print the ASR summary for an existing results CSV without running any trials")
    args = parser.parse_args()

    if args.summarize:
        print_asr_summary(load_results_csv(args.summarize))
        raise SystemExit

    deliveries = DELIVERY_MODES if args.delivery == "both" else (args.delivery,)
    print(f"Running {args.trials} trials per scenario ({len(SUITES[args.suite])} scenarios in suite '{args.suite}' x "
          f"{len(deliveries)} delivery) -- gate: {args.gate}, delivery: {args.delivery}\n")

    if args.resume:
        csv_path = args.resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_tag = "" if args.suite == "text" else f"{args.suite}_"
        csv_path = f"{RESULTS_DIR}/trials_{args.gate}_{suite_tag}{args.delivery}_{timestamp}.csv"
    print(f"Writing each trial to: {csv_path}\n")

    start = time.time()
    try:
        results = run_all_trials(args.trials, csv_path, gate_label=args.gate, keep_chroma=args.keep_chroma,
                                 deliveries=deliveries, resume=bool(args.resume), suite=args.suite)
    except DailyLimitReached as e:
        print(f"\n\nStopped: Groq daily quota exhausted.\n  {str(e)[:200]}")
        print(f"\nFinished trials are saved in {csv_path}")
        print(f"After the quota resets, rerun the same command with:  --resume {csv_path}")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Finished trials are saved in {csv_path}")
        print(f"Continue by rerunning the same command with:  --resume {csv_path}")
        raise SystemExit(1)
    elapsed = time.time() - start

    print_asr_summary(results)
    print(f"\nTotal time: {elapsed:.0f}s")
