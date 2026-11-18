#!/usr/bin/env python3
"""$ per 1,000 prior-auth summaries: self-hosted NIM (any cloud) vs. a per-token API, from YOUR measurements.
🔁 Lab 02/03's cost model, unchanged on purpose: the formula is the same on every cloud - only the inputs move.

Inputs come from the M5 sweep: the highest request rate that still meets the SLO (the operating point).
API prices and NVIDIA AI Enterprise terms are YOUR inputs (check the current price sheet) - nothing is hard-coded.
Example (AKS A100 Spot, long-prompt summaries, 50% duty cycle, NVAIE via marketplace at $1/GPU-hour):
  python3 bench/cost_model.py --node-hourly 0.6788 --nvaie-hourly 1.0 --req-per-s 4.2 \
      --isl 1400 --osl 300 --duty-cycle 0.5 --api-in-per-1m 0.5 --api-out-per-1m 1.5
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node-hourly", type=float, required=True, help="VM $/hour (Spot or PAYG or reserved)")
    ap.add_argument("--slice-share", type=float, default=1.0, help="tenant share of the GPU (3g slice = 3/7)")
    ap.add_argument("--nvaie-hourly", type=float, default=0.0, help="NVIDIA AI Enterprise $/GPU-hour (marketplace) - 0 if BYOL")
    ap.add_argument("--req-per-s", type=float, required=True, help="SLO-compliant throughput from the sweep")
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--duty-cycle", type=float, default=1.0, help="fraction of paid hours the capacity is actually busy")
    ap.add_argument("--api-in-per-1m", type=float, help="API $ per 1M input tokens (your input)")
    ap.add_argument("--api-out-per-1m", type=float, help="API $ per 1M output tokens (your input)")
    a = ap.parse_args()

    hourly = (a.node_hourly + a.nvaie_hourly) * a.slice_share
    per_hour_capacity = a.req_per_s * 3600
    busy = per_hour_capacity * a.duty_cycle
    self_1k = hourly / busy * 1000
    print(f"Self-hosted capacity: {per_hour_capacity:,.0f} summaries/hour at the SLO operating point")
    print(f"Self-hosted cost:     ${self_1k:,.2f} per 1,000 summaries at {a.duty_cycle:.0%} duty cycle "
          f"(${hourly / per_hour_capacity * 1000:,.2f} if always busy)")
    if a.api_in_per_1m is not None and a.api_out_per_1m is not None:
        api_decision = (a.isl * a.api_in_per_1m + a.osl * a.api_out_per_1m) / 1e6
        print(f"API cost:             ${api_decision * 1000:,.2f} per 1,000 summaries")
        breakeven = hourly / api_decision
        print(f"Break-even volume:    {breakeven:,.0f} summaries/hour ({breakeven / per_hour_capacity:.0%} of capacity). "
              f"Above that, self-hosting is cheaper; below it, the API is.")
    print("Remember the non-GPU costs: system nodes, storage, observability, and the engineers who run it.")


if __name__ == "__main__":
    main()
