#!/usr/bin/env python3
"""M11 weekly technical sync pack: status, releases since last week, what's new upstream, open issues, decisions.
The JD calls it "regular technical meetings with customers covering project updates, new features, and
troubleshooting" - this makes that meeting a 30-minute habit instead of a scramble.
  python3 fleet/fleet_report.py --out results/fleet.md && python3 fleet/whats_new.py --json results/whats-new.json
  python3 fleet/weekly_sync.py --fleet results/fleet.json --whats-new results/whats-new.json --out results/weekly-sync.md
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def releases_since(days):
    """Releases = commits to clusters/*/values.yaml (canary / promote / rollback) - GitOps makes this free."""
    r = subprocess.run(["git", "-C", str(ROOT), "log", f"--since={days}.days", "--pretty=format:%ad %s", "--date=short",
                        "--", "clusters/"], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleet", default=None, help="fleet_report.py --out <x>.md writes <x>.json")
    ap.add_argument("--whats-new", default=None)
    ap.add_argument("--actions", default=str(ROOT / "fleet" / "actions.yaml"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fleet = json.loads(Path(a.fleet).read_text()) if a.fleet else []
    news = json.loads(Path(a.whats_new).read_text()) if a.whats_new else []
    actions = yaml.safe_load(Path(a.actions).read_text()) or []
    L = [f"# Plover x NVIDIA - weekly technical sync ({datetime.now(timezone.utc).date().isoformat()})", "",
         "**Agenda (30 min):** status 5 - releases 5 - new features 5 - issues 10 - decisions 5", "", "## 1. Status at a glance"]
    for s in fleet:
        flag = "attention" if s["attention"] else "green"
        L.append(f"- **{s['cluster']}** ({', '.join(s['gpu_products'])}): {flag}; ${s['usd_per_hour']:.2f}/h; "
                 + ("; ".join(s["attention"][:3]) if s["attention"] else "synced, healthy, pinned NIM"))
    if not fleet:
        L.append("- (run fleet/fleet_report.py)")
    L += ["", f"## 2. Releases in the last {a.days} days (Git = release log)"]
    L += [f"- {r}" for r in releases_since(a.days)] or ["- none"]
    L += ["", "## 3. What's new upstream that affects you"]
    behind = [n for n in news if n["status"] not in ("current",) and not n["status"].startswith("unknown")]
    L += [f"- {n['pin'].replace('PIN_', '').lower()}: {n['pinned']} -> {n['latest']} ({n['status']}) - notes: {n['notes']}" for n in behind] \
        or ["- Everything pinned is current."]
    L += ["- Model news: Nemotron 3.5 Super VL + Bolt (2026-10-15) - see addon-nemotron35/customer-brief.md"]
    L += ["", "## 4. Open issues and troubleshooting"]
    for x in actions:
        if x.get("status") != "done" and x.get("kind") != "decision":
            L.append(f"- [{x['id']}] {x['title']} - owner {x['owner']}, due {x['due']} ({x['kind']})")
    L += ["", "## 5. Decisions needed"]
    L += [f"- [{x['id']}] {x['title']} - owner {x['owner']}, due {x['due']}" for x in actions
          if x.get("status") != "done" and x.get("kind") == "decision"] or ["- none"]
    for s in fleet:
        if s.get("routing") and s["routing"].get("canary"):
            L.append(f"- Canary on {s['cluster']}: promote {s['routing']['canary']} or roll back (eval gate result in MLflow)")
    L += ["", "## 6. Next steps (owner, date)", "- ", ""]
    md = "\n".join(L)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
