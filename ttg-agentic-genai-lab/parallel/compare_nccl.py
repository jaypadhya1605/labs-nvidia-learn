#!/usr/bin/env python3
"""M7 - read nccl-tests output and compare it with the link you THINK you have (🔁 Lab 01: measure, then model).

nccl-tests prints two bandwidths per message size:
  algbw = bytes / time                        (what one GPU's buffer moved)
  busbw = algbw x 2(n-1)/n for all_reduce     (normalised so it is comparable to the link rate for any n)
Compare busbw on large messages with the per-direction link rate; read latency from the smallest messages.

Usage (from the repo root):
  python3 -m parallel.compare_nccl results/nccl/<run>/all_reduce_perf.log --link nvlink4
  python3 -m parallel.compare_nccl results/nccl/<run>/all_reduce_perf.log --link pcie4 --gpus 4
  python3 -m parallel.compare_nccl a.log b.log --link nvlink4         # side by side: NVLS on vs off, P2P on vs off
  python3 parallel/compare_nccl.py results/nccl/                      # a directory: every *_perf.log inside
  python3 -m parallel.compare_nccl parallel/samples/all_reduce_perf_format_example.txt --link nvlink4  # format demo
Plug the result back into parallel/parallelism_calc.py: EFFICIENCY = busbw / link rate; alpha = small-message time.
"""
import argparse
import glob
import json
import os
import re
import sys

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parallel.parallelism_calc import LINKS  # noqa: E402

BUS_FACTORS = {  # busbw / algbw for each collective (nccl-tests doc/PERFORMANCE.md)
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def parse(text, name=""):
    """Rows of an nccl-tests *_perf log (both the old 'error' and the new '#wrong' column layouts).

    The collective comes from the log text when it names it, else from the file name (all_reduce_perf.log).
    """
    ngpus = None
    m = re.search(r"nGpus\s+(\d+)", text)
    if m:
        ngpus = int(m.group(1))
    ranks = len(re.findall(r"^#\s+Rank\s+\d+", text, re.M))
    if ranks and (ngpus is None or ranks > ngpus):
        ngpus = ranks
    collective = None
    for src in (text, name):
        m = re.search(r"(all_reduce|all_gather|reduce_scatter|alltoall|broadcast|reduce)_perf", src or "")
        if m:
            collective = m.group(1)
            break
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 12 or not parts[0].isdigit():
            continue
        try:
            size, count = int(parts[0]), int(parts[1])
            nums = [float(x) for x in parts[5:13]]
        except ValueError:
            continue
        rows.append({"size": size, "count": count, "type": parts[2], "redop": parts[3],
                     "oop_time_us": nums[0], "oop_algbw": nums[1], "oop_busbw": nums[2],
                     "ip_time_us": nums[4], "ip_algbw": nums[5], "ip_busbw": nums[6]})
    avg = None
    m = re.search(r"Avg bus bandwidth\s*:\s*([0-9.]+)", text)
    if m:
        avg = float(m.group(1))
    return {"ngpus": ngpus, "collective": collective, "rows": rows, "avg_busbw": avg}


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:,.0f} {unit}" if unit == "B" else f"{nbytes:,.0f} {unit}"
        nbytes /= 1024
    return str(nbytes)


def summarize(parsed, link, gpus=None):
    rows = parsed["rows"]
    if not rows:
        raise ValueError("no result rows found - is this an nccl-tests *_perf output?")
    n = gpus or parsed["ngpus"] or 8
    best = max(rows, key=lambda r: max(r["oop_busbw"], r["ip_busbw"]))
    peak = max(best["oop_busbw"], best["ip_busbw"])
    knee = next((r for r in rows if max(r["oop_busbw"], r["ip_busbw"]) >= 0.9 * peak), best)
    small = rows[0]
    lk = LINKS[link]
    factor = BUS_FACTORS.get(parsed["collective"] or "all_reduce", BUS_FACTORS["all_reduce"])(n)
    consistency = [abs(r["oop_algbw"] * factor - r["oop_busbw"]) <= 0.02 * max(r["oop_busbw"], 0.01) + 0.011
                   for r in rows if r["oop_busbw"] > 1]
    return {"collective": parsed["collective"] or "all_reduce", "gpus": n, "link": link,
            "link_gbs": lk["gbs"], "peak_busbw_gbs": peak, "peak_at_bytes": best["size"],
            "efficiency": peak / lk["gbs"], "bandwidth_regime_from_bytes": knee["size"],
            "small_message_us": min(small["oop_time_us"], small["ip_time_us"]), "avg_busbw_gbs": parsed["avg_busbw"],
            "bus_factor": factor, "busbw_consistent": all(consistency) if consistency else None}


def print_report(path, parsed, s):
    print(f"## {os.path.basename(path)} - {s['collective']} on {s['gpus']} GPUs vs {s['link']} "
          f"({s['link_gbs']} GB/s per direction)\n")
    pick = [r for r in parsed["rows"] if r["size"] in (8, 1024, 65536, 1 << 20, 16 << 20, 64 << 20, 256 << 20,
                                                      1 << 30, 4 << 30)] or parsed["rows"]
    print("| size | time us | algbw GB/s | busbw GB/s | % of link |")
    print("|---|---|---|---|---|")
    for r in pick:
        bus = max(r["oop_busbw"], r["ip_busbw"])
        print(f"| {human(r['size'])} | {min(r['oop_time_us'], r['ip_time_us']):,.1f} | {max(r['oop_algbw'], r['ip_algbw']):,.2f} "
              f"| {bus:,.2f} | {bus / s['link_gbs']:.0%} |")
    print()
    print(f"- Peak bus bandwidth {s['peak_busbw_gbs']:,.1f} GB/s at {human(s['peak_at_bytes'])} = "
          f"{s['efficiency']:.0%} of the {s['link']} rate -> use EFFICIENCY = {min(s['efficiency'], 1.0):.2f}.")
    print(f"- Bandwidth regime starts around {human(s['bandwidth_regime_from_bytes'])} (90% of peak); below that "
          f"you pay latency, not bytes.")
    print(f"- Smallest message took {s['small_message_us']:,.1f} us -> use alpha ~ {s['small_message_us']:,.0f} us "
          f"for decode-time TP all-reduces (they are tiny: batch x hidden x 2 bytes).")
    if s["busbw_consistent"] is False:
        print(f"- WARNING: busbw is not algbw x {s['bus_factor']:.3f}; check --gpus (busbw factor assumes n = {s['gpus']}).")
    if s["avg_busbw_gbs"] is not None:
        print(f"- nccl-tests 'Avg bus bandwidth' (all sizes): {s['avg_busbw_gbs']:,.2f} GB/s - an average over tiny "
              f"and huge messages; quote the large-message number instead.")
    print()


def expand(paths):
    """Files as given; directories -> every *_perf.log below them (all_reduce first), so results/nccl/ works."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*_perf.log"), recursive=True),
                           key=lambda f: (os.path.dirname(f), "all_reduce" not in os.path.basename(f), f))
            if not found:
                raise SystemExit(f"no *_perf.log files under {p} (run parallel/nccl_tests.sh first)")
            out += found
        else:
            out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="nccl-tests output files, or directories (every *_perf.log inside)")
    ap.add_argument("--link", default="nvlink4", choices=sorted(LINKS))
    ap.add_argument("--gpus", type=int, default=None, help="override the GPU count found in the log")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = []
    for path in expand(a.logs):
        with open(path) as f:
            parsed = parse(f.read(), os.path.basename(path))
        s = summarize(parsed, a.link, a.gpus)
        out.append({"file": path, **s})
        if not a.json:
            print_report(path, parsed, s)
    if a.json:
        print(json.dumps(out, indent=2))
    elif len(out) > 1:
        print("| file | peak busbw GB/s | % of link | small-message us |")
        print("|---|---|---|---|")
        for s in out:
            print(f"| {os.path.basename(s['file'])} | {s['peak_busbw_gbs']:,.1f} | {s['efficiency']:.0%} | "
                  f"{s['small_message_us']:,.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
