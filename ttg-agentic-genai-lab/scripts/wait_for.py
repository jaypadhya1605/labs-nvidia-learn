#!/usr/bin/env python3
"""Wait until every URL answers HTTP (any status counts: /mcp answers 405/406 to a GET and that still means "up").

  python scripts/wait_for.py http://localhost:8000/v1/models http://localhost:8101/mcp --timeout 90

Exit 0 when all are up, 1 on timeout. Standard library only (CI images like python:3.12-slim have no curl).
"""
import argparse
import sys
import time
import urllib.error
import urllib.request


def is_up(url: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - local health checks only
        return True
    except urllib.error.HTTPError:
        return True          # the server answered: it is listening
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def wait(urls: list[str], timeout: float = 90.0, interval: float = 1.0) -> list[str]:
    """Returns the URLs still down after `timeout` seconds (empty list = all up)."""
    deadline = time.monotonic() + timeout
    pending = list(urls)
    while pending and time.monotonic() < deadline:
        pending = [u for u in pending if not is_up(u)]
        if pending:
            time.sleep(interval)
    return pending


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args(argv)
    down = wait(args.urls, args.timeout)
    for u in args.urls:
        print(f"  [{'DOWN' if u in down else 'up':4s}] {u}")
    return 1 if down else 0


if __name__ == "__main__":
    sys.exit(main())
