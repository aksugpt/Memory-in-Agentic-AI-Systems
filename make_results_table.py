"""
make_results_table.py

Builds the thesis results table from every results/trials_*.csv:
one row per (configuration, suite, delivery), with

  ASR          attack successes / attack trials, Wilson 95% CI
  p vs none    two-sided Fisher exact test against no_gate on the
               same suite + delivery (does this config differ from
               the undefended agent?)
  utility      benign scenarios that executed / benign trials
  uncorr-auto  benign scenarios whose value is untrusted and NOT
               independently corroborated, that executed WITHOUT
               asking the user (lower is safer; TMA-NM's metric)

Errored trials are excluded, as in run_trials.py.

By default each group uses ONE file: the most complete one (most
non-errored trials; latest if tied). Files are never pooled, so a
smoke test or a run of older code doesn't get silently mixed in, and
a run cut short by the daily quota doesn't replace a complete one.
When older files exist for a group they're listed underneath, so a
disagreement (like the two no_gate upload baselines) stays visible.
--all-files shows every file as its own row instead.

Usage:
    python make_results_table.py              # markdown to stdout + results/summary_table.{md,csv}
    python make_results_table.py --detail     # also per-scenario rows
    python make_results_table.py --all-files
"""

import argparse
import csv
import glob
import os
import re

from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint

from attacks.indirect_injection import ACTION_SCENARIOS, POISON_SCENARIOS
from run_trials import CONFIGS, load_results_csv

RESULTS_DIR = "./results"
TEXT_NAMES = {s["name"] for s in POISON_SCENARIOS}
CONFIRM_NAMES = {s["name"] for s in ACTION_SCENARIOS if s.get("user_confirms")}
CONFIG_ORDER = {name: i for i, name in enumerate(CONFIGS)}


def _timestamp(path: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else ""


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "--"
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return f"{k}/{n} = {k / n:.1%} [{low:.1%}, {high:.1%}]"


def load_groups(all_files: bool) -> tuple[dict, dict]:
    """
    Returns (groups, older): groups[key] = (file, rows); older[key] =
    [(file, rows), ...] for the files not used. key = (gate, suite,
    delivery, file-if-all_files).
    """
    by_key: dict[tuple, list] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "trials_*.csv")), key=_timestamp):
        for r in load_results_csv(path):
            if r["error"]:
                continue
            suite = "text" if r["scenario"] in TEXT_NAMES else "action"
            key = (r["gate"], suite, r["delivery"])
            by_key.setdefault(key, {}).setdefault(path, []).append(r)

    groups, older = {}, {}
    for key, files in by_key.items():
        paths = sorted(files, key=lambda p: (len(files[p]), _timestamp(p)))
        if all_files:
            for p in paths:
                groups[key + (os.path.basename(p),)] = (p, files[p])
        else:
            groups[key] = (paths[-1], files[paths[-1]])
            older[key] = [(p, files[p]) for p in paths[:-1]]
    return groups, older


def stats(rows: list[dict]) -> dict:
    attack = [r for r in rows if r.get("kind", "attack") == "attack"]
    benign = [r for r in rows if r.get("kind") == "benign"]
    confirm = [r for r in benign if r["scenario"] in CONFIRM_NAMES]
    return {
        "asr_k": sum(r["trigger_succeeded"] for r in attack), "asr_n": len(attack),
        "util_k": sum(r["trigger_succeeded"] for r in benign), "util_n": len(benign),
        "auto_k": sum(bool(r.get("auto_executed")) for r in confirm), "auto_n": len(confirm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", action="store_true", help="Also list every scenario within each group")
    parser.add_argument("--all-files", action="store_true", help="One row per results file instead of latest-per-group")
    args = parser.parse_args()

    groups, older = load_groups(args.all_files)
    if not groups:
        print(f"No results in {RESULTS_DIR}/")
        return

    def sort_key(key):
        gate, suite, delivery = key[:3]
        return (suite != "text", delivery, CONFIG_ORDER.get(gate, 99), key[3:] if len(key) > 3 else ())

    header = ["suite", "delivery", "config", "ASR (attacks)", "p vs no_gate", "legit utility", "uncorr-auto", "file"]
    table, notes = [], []
    for key in sorted(groups, key=sort_key):
        gate, suite, delivery = key[:3]
        path, rows = groups[key]
        s = stats(rows)

        p_value = "--"
        baseline = groups.get(("no_gate", suite, delivery) + key[3:]) if not args.all_files else None
        if baseline is None and args.all_files:
            candidates = [v for k, v in groups.items() if k[:3] == ("no_gate", suite, delivery)]
            baseline = candidates[-1] if candidates else None
        if gate != "no_gate" and baseline is not None and s["asr_n"]:
            b = stats(baseline[1])
            if b["asr_n"]:
                _, p = fisher_exact([[s["asr_k"], s["asr_n"] - s["asr_k"]], [b["asr_k"], b["asr_n"] - b["asr_k"]]])
                p_value = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"

        table.append([suite, delivery, gate, _rate(s["asr_k"], s["asr_n"]), p_value,
                      _rate(s["util_k"], s["util_n"]), _rate(s["auto_k"], s["auto_n"]), os.path.basename(path)])

        if args.detail:
            for name in sorted({r["scenario"] for r in rows}):
                sub = [r for r in rows if r["scenario"] == name]
                k = sum(r["trigger_succeeded"] for r in sub)
                kind = sub[0].get("kind", "attack")
                table.append(["", "", f"  - {name} ({kind})", _rate(k, len(sub)) if kind == "attack" else "",
                              "", _rate(k, len(sub)) if kind == "benign" else "", "", ""])

        for p, old_rows in older.get(key, []):
            o = stats(old_rows)
            notes.append(f"- {gate} / {suite} / {delivery}: older file {os.path.basename(p)} "
                         f"had ASR {_rate(o['asr_k'], o['asr_n'])} (not used -- fewer valid trials; see --all-files)")

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in table]
    if notes:
        md += ["", "Older runs of the same group (excluded above):"] + notes
    md += ["", "ASR: Wilson 95% CI. p: two-sided Fisher exact vs no_gate, same suite + delivery. "
           "Errored trials excluded. uncorr-auto: untrusted, uncorroborated legit value executed without asking (lower is safer)."]
    text = "\n".join(md)
    print(text)

    with open(os.path.join(RESULTS_DIR, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS_DIR, "summary_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"\nSaved {RESULTS_DIR}/summary_table.md and summary_table.csv")


if __name__ == "__main__":
    main()
