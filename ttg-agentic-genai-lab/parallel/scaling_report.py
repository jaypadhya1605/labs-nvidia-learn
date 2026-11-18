#!/usr/bin/env python3
"""M7 - turn parallel/train_scaling.sh logs into a scaling table (samples/s, speedup, efficiency, peak memory).

Reads <run>.log (Hugging Face Trainer / TRL final metrics: train_samples_per_second, train_runtime) and
<run>.mem.csv (nvidia-smi timestamp,index,memory.used MiB samples) for every run in a directory.

Usage: python3 -m parallel.scaling_report results/scaling/            # newest run under it
       python3 parallel/scaling_report.py results/scaling/<timestamp> [--json]
Scaling efficiency = (samples/s on N GPUs) / (N x samples/s on 1 GPU). Weak scaling: per-GPU batch fixed.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

METRIC = r"['\"]?{}['\"]?\s*[:=]\s*([0-9][0-9:.eE+-]*)"


def to_number(v):
    """'123.4' -> 123.4; Trainer.log_metrics prints runtimes as H:MM:SS.ss -> seconds."""
    if ":" in v:
        sec = 0.0
        for part in v.split(":"):
            sec = sec * 60 + float(part)
        return sec
    return float(v)


def parse_log(text):
    """Final Hugging Face Trainer metrics: {'train_runtime': 81.2, 'train_samples_per_second': 11.8, ...}."""
    out = {}
    for key in ("train_samples_per_second", "train_steps_per_second", "train_runtime", "train_loss"):
        hits = re.findall(METRIC.format(key), text)
        if hits:
            out[key] = to_number(hits[-1].rstrip(".,"))     # the final summary line wins
    return out


def parse_mem(path):
    """Peak MiB per GPU index from an nvidia-smi CSV."""
    peak = {}
    if not os.path.exists(path):
        return peak
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                idx, mib = int(row[1].strip()), float(row[2].strip())
            except ValueError:
                continue
            peak[idx] = max(peak.get(idx, 0.0), mib)
    return peak


def run_info(name):
    m = re.match(r"(single|ddp|fsdp2)-(\d+)gpu", name)
    if not m:
        return None
    return {"strategy": m.group(1), "gpus": int(m.group(2))}


def run_dir(directory):
    """The directory with the run logs: `directory` itself, else its newest subdirectory that has them
    (so results/scaling/ picks the latest results/scaling/<timestamp>/)."""
    def has_logs(d):
        return any(run_info(os.path.splitext(os.path.basename(f))[0]) for f in glob.glob(os.path.join(d, "*.log")))
    if has_logs(directory):
        return directory
    subs = [d for d in glob.glob(os.path.join(directory, "**", ""), recursive=True) if has_logs(d)]
    return max(subs, key=os.path.getmtime) if subs else directory


def report(directory):
    directory = run_dir(directory)
    runs = []
    for log in sorted(glob.glob(os.path.join(directory, "*.log"))):
        name = os.path.splitext(os.path.basename(log))[0]
        info = run_info(name)
        if not info:
            continue
        with open(log, errors="ignore") as f:
            metrics = parse_log(f.read())
        peak = parse_mem(os.path.join(directory, f"{name}.mem.csv"))
        runs.append({"run": name, **info, **metrics,
                     "peak_mem_gib": max(peak.values()) / 1024 if peak else None})
    base = next((r for r in runs if r["gpus"] == 1 and r.get("train_samples_per_second")), None)
    for r in runs:
        sps = r.get("train_samples_per_second")
        if sps:
            r["samples_per_s_per_gpu"] = sps / r["gpus"]
        if base and sps:
            r["speedup"] = sps / base["train_samples_per_second"]
            r["efficiency"] = r["speedup"] / r["gpus"]
    return sorted(runs, key=lambda r: (r["gpus"], r["strategy"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    runs = report(a.directory)
    if a.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print(f"no single-/ddp-/fsdp2-<n>gpu logs in {a.directory}")
        return 1
    print(f"Runs in {run_dir(a.directory)}\n")
    print("| run | GPUs | strategy | samples/s | samples/s per GPU | speedup | scaling efficiency | peak GiB per GPU | "
          "runtime s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        def f(k, fmt):
            return fmt.format(r[k]) if r.get(k) is not None else "-"
        print(f"| {r['run']} | {r['gpus']} | {r['strategy']} | {f('train_samples_per_second', '{:,.2f}')} | "
              f"{f('samples_per_s_per_gpu', '{:,.2f}')} | {f('speedup', '{:.2f}x')} | {f('efficiency', '{:.0%}')} | {f('peak_mem_gib', '{:,.1f}')} | "
              f"{f('train_runtime', '{:,.0f}')} |")
    print("\nDDP keeps a full model copy per GPU (memory flat, gradients all-reduced); FSDP2 shards weights, "
          "gradients and optimizer states (memory drops with N, plus all-gather/reduce-scatter traffic). "
          "For a LoRA job the trainable state is tiny, so expect DDP and FSDP2 to scale alike - FSDP2 wins on "
          "memory only when the FROZEN base model is the problem (70B on 80 GB GPUs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
