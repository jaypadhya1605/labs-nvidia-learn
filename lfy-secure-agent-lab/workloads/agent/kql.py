"""Minimal Kusto REST client (standard library only) for the Kustainer emulator - used by the loader and the agent tools."""
import json
import urllib.request


def _post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def mgmt(base, db, csl, timeout=60):
    """Control command (.create table, .ingest inline ...) via the v1 management endpoint."""
    return _post(base.rstrip("/") + "/v1/rest/mgmt", {"db": db, "csl": csl}, timeout)


def parse_v2(frames):
    """Return the PrimaryResult table of a v2 query response as a list of dicts."""
    for f in frames:
        if f.get("FrameType") == "DataTable" and f.get("TableKind") == "PrimaryResult":
            cols = [c["ColumnName"] for c in f["Columns"]]
            return [dict(zip(cols, row)) for row in f["Rows"] if isinstance(row, list)]
    return []


def query(base, db, csl, timeout_s=30, max_records=500):
    props = {"Options": {"servertimeout": f"00:00:{min(timeout_s, 59):02d}", "truncationmaxrecords": max_records}}
    frames = _post(base.rstrip("/") + "/v2/rest/query", {"db": db, "csl": csl, "properties": json.dumps(props)}, timeout_s + 5)
    return parse_v2(frames)
