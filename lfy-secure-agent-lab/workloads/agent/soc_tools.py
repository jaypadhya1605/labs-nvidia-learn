"""SOC tools with policy enforced AT THE TOOL BOUNDARY: the model proposes, code decides.
Shared by the transparent Python agent (agent.py) and the MCP server used by NeMo Agent Toolkit (mcp_server.py)."""
import json
import math
import os
import re
import urllib.request
from pathlib import Path

import kql

KUSTO = os.environ.get("KUSTO_URL", "http://kustainer.lfy-data.svc:8080")
DB = "NetDefaultDB"
EMBED_URL = os.environ.get("EMBED_URL", "http://agent-embed.lfy-agent.svc:8000/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
PROTECTED_ACCOUNTS = {"soc-admin@kha.example", "breakglass@kha.example"}
ALLOWED_ACTIONS = {"disable_account", "isolate_host", "block_ip", "revoke_sessions"}
MAX_CHARS = 6000
KQL_DENY = [r"^\s*\.", r"\bexternaldata\b", r"\bevaluate\b", r"\bhttp_request", r"\bcluster\s*\(", r"\bdatabase\s*\(",
            r"\.(set|append|drop|ingest|alter|delete|purge)\b"]
IPV4 = re.compile(r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$")


class PolicyDenied(Exception):
    pass


def truncate(obj):
    s = json.dumps(obj, default=str)
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + '..."[truncated]"'


def check_kql(query):
    for pat in KQL_DENY:
        if re.search(pat, query, re.I | re.M):
            raise PolicyDenied(f"KQL policy: '{pat}' is not allowed (read-only queries, no plugins, no cross-cluster, no control commands)")
    if len(query) > 2000:
        raise PolicyDenied("KQL policy: query longer than 2000 characters")
    return query.strip().rstrip(";") + "\n| take 50"


def run_kql(query):
    return truncate(kql.query(KUSTO, DB, check_kql(query), timeout_s=20))


def lookup_ip(ip):
    if not IPV4.match(ip or ""):
        raise PolicyDenied("lookup_ip accepts a single IPv4 address only")  # never splice free text into a query
    rows = kql.query(KUSTO, DB, f"ThreatIntel | where IPAddress == '{ip}'")
    return truncate(rows or [{"IPAddress": ip, "Reputation": "unknown"}])


def _embed(texts, input_type):
    body = json.dumps({"model": EMBED_MODEL, "input": texts, "input_type": input_type}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return [d["embedding"] for d in json.load(r)["data"]]


def _cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    return num / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


class Playbooks:
    """Retrieval over playbook files. Defended mode loads ONLY the signature-verified bundle."""

    def __init__(self, dirs):
        self.docs = []
        for d in dirs:
            for p in sorted(Path(d).glob("*.md")) if d and Path(d).is_dir() else []:
                self.docs.append((p.name, p.read_text()))
        self.vecs = None

    def search(self, question, k=2):
        if not self.docs:
            return "[]"
        try:
            if self.vecs is None:
                self.vecs = _embed([t for _, t in self.docs], "passage")
            q = _embed([question], "query")[0]
            ranked = sorted(zip(self.docs, self.vecs), key=lambda dv: -_cos(q, dv[1]))
            hits = [d for d, _ in ranked[:k]]
        except Exception:  # embedding NIM unavailable -> keyword fallback keeps the agent working
            words = set(re.findall(r"[a-z]{4,}", question.lower()))
            hits = sorted(self.docs, key=lambda d: -len(words & set(re.findall(r"[a-z]{4,}", d[1].lower()))))[:k]
        return truncate([{"playbook": n, "text": t} for n, t in hits])


def propose_action(action, target, reason, defended=True):
    if action not in ALLOWED_ACTIONS:
        raise PolicyDenied(f"unknown action '{action}'")
    if defended:
        if str(target).lower() in PROTECTED_ACCOUNTS:
            raise PolicyDenied("privileged and break-glass accounts are never actioned by automation")
        return {"action": action, "target": target, "status": "PENDING_HUMAN_APPROVAL", "reason": reason}
    return {"action": action, "target": target, "status": "EXECUTED (simulated)", "reason": reason}
