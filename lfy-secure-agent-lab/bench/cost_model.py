#!/usr/bin/env python3
"""$ per 1,000 decisions: self-hosted NIM on a (slice of a) GPU vs. a per-token API, from YOUR measurements.

Inputs come from the M7 sweep: the highest request rate that still meets the SLO (the operating point).
API prices are YOUR inputs (check the vendor's current price sheet) - nothing is hard-coded.
Example:
  python3 bench/cost_model.py --node-hourly 0.6788 --slice-share 0.4286 --req-per-s 0.9 \
      --isl 6000 --osl 350 --duty-cycle 0.5 --api-in-per-1m 3 --api-out-per-1m 15
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node-hourly", type=float, required=True, help="VM $/hour (Spot or PAYG or reserved)")
    ap.add_argument("--slice-share", type=float, default=1.0, help="tenant share of the GPU (3g slice = 3/7)")
    ap.add_argument("--req-per-s", type=float, required=True, help="SLO-compliant throughput from the sweep")
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--duty-cycle", type=float, default=1.0, help="fraction of paid hours the capacity is actually busy")
    ap.add_argument("--api-in-per-1m", type=float, help="API $ per 1M input tokens (your input)")
    ap.add_argument("--api-out-per-1m", type=float, help="API $ per 1M output tokens (your input)")
    a = ap.parse_args()

    hourly = a.node_hourly * a.slice_share
    per_hour_capacity = a.req_per_s * 3600
    busy = per_hour_capacity * a.duty_cycle
    self_1k = hourly / busy * 1000
    print(f"Self-hosted capacity: {per_hour_capacity:,.0f} decisions/hour at the SLO operating point")
    print(f"Self-hosted cost:     ${self_1k:,.2f} per 1,000 decisions at {a.duty_cycle:.0%} duty cycle "
          f"(${hourly / per_hour_capacity * 1000:,.2f} if always busy)")
    if a.api_in_per_1m is not None and a.api_out_per_1m is not None:
        api_decision = (a.isl * a.api_in_per_1m + a.osl * a.api_out_per_1m) / 1e6
        print(f"API cost:             ${api_decision * 1000:,.2f} per 1,000 decisions")
        breakeven = hourly / api_decision
        print(f"Break-even volume:    {breakeven:,.0f} decisions/hour ({breakeven / per_hour_capacity:.0%} of capacity). "
              f"Above that, self-hosting is cheaper; below it, the API is.")
    print("Remember the non-GPU costs: system nodes, storage, observability, and the engineers who run it.")


if __name__ == "__main__":
    main()
