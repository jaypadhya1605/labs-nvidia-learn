#!/usr/bin/env python3
"""Refresh selector prices from the clouds' PUBLIC price feeds (no credentials) and show what changed.
  Azure: Retail Prices API      AWS: public EC2 price feed (gzip JSON)      OCI: public price list
  Google: no keyless JSON feed - update instances.yaml from cloud.google.com/products/compute/pricing/accelerator-optimized
Usage: python3 selector/fetch_prices.py [--write]
"""
import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AWS_FEED = ("https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/"
            "US%20East%20(N.%20Virginia)/Linux/index.json")
OCI_FEED = "https://apexapps.oracle.com/pls/apex/cetools/api/v1/products/?currencyCode=USD"
OCI_PARTS = {"VM.GPU.A10.1": ("B95909", 1), "BM.GPU.H100.8": ("B98415", 8)}   # part number, GPUs priced per GPU-hour


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        except Exception:  # noqa: BLE001 - public feeds rate-limit (Azure 429); back off and retry
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"could not fetch {url}")


def azure(sku, region):
    f = f"armRegionName eq '{region}' and armSkuName eq '{sku}' and priceType eq 'Consumption'"
    items = get("https://prices.azure.com/api/retail/prices?" + urllib.parse.urlencode({"$filter": f})).get("Items", [])
    od = [i["retailPrice"] for i in items if "Windows" not in i["productName"] and "Spot" not in i["meterName"] and "Low Priority" not in i["meterName"]]
    sp = [i["retailPrice"] for i in items if "Windows" not in i["productName"] and "Spot" in i["meterName"]]
    return (min(od) if od else None, min(sp) if sp else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update instances.yaml in place")
    a = ap.parse_args()
    insts = yaml.safe_load((HERE / "instances.yaml").read_text())
    aws = None
    oci = None
    for inst in insts:
        old = (inst["on_demand"], inst.get("spot"))
        try:
            if inst["cloud"] == "azure":
                od, sp = azure(inst["instance"], inst["region"])
                time.sleep(3)
            elif inst["cloud"] == "aws":
                aws = aws or get(AWS_FEED)
                hits = [v for reg in aws["regions"].values() for v in reg.values() if v.get("Instance Type") == inst["instance"]]
                od, sp = (float(hits[0]["price"]) if hits else None), inst.get("spot")
            elif inst["cloud"] == "oci" and inst["instance"] in OCI_PARTS:
                oci = oci or get(OCI_FEED)
                part, n = OCI_PARTS[inst["instance"]]
                item = [i for i in oci["items"] if i.get("partNumber") == part][0]
                usd = [p for p in item["currencyCodeLocalizations"] if p["currencyCode"] == "USD"][0]["prices"][0]["value"]
                od, sp = float(usd) * n, None
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  {inst['instance']:<28} skipped: {exc}")
            continue
        if od is not None:
            inst["on_demand"] = round(od, 6)
        if sp is not None:
            inst["spot"] = round(sp, 6)
        if old != (inst["on_demand"], inst.get("spot")):
            print(f"  {inst['instance']:<28} {old} -> {(inst['on_demand'], inst.get('spot'))}")
        else:
            print(f"  {inst['instance']:<28} unchanged {old}")
    if a.write:
        (HERE / "instances.yaml").write_text(yaml.safe_dump(insts, sort_keys=False, width=200))
        print("instances.yaml updated - also update the 'Retrieved' date in its header")


if __name__ == "__main__":
    main()
