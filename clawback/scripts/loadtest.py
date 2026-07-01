#!/usr/bin/env python3
"""
Dependency-free load test for Clawback — stress the API locally before putting
it in front of users (the step everyone skips). Stdlib only; no k6/locust needed.

Start the app (./run.sh) in one terminal, then e.g.:
    python scripts/loadtest.py --url http://localhost:8000 -n 500 -c 50

429s in the output are expected and good — that's the rate limiter doing its
job. To benchmark raw throughput instead, start the app with a high limit:
    CLAWBACK_RATE_LIMIT=1000000 ./run.sh
"""
import time
import json
import argparse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAYLOAD = json.dumps({
    "scenario": "flight", "region": "za", "company": "LoadTest Air",
    "date": "1 June 2026", "amount": "R1000",
}).encode()


def one(url: str):
    req = urllib.request.Request(
        url + "/api/preview", data=PAYLOAD,
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception:
        code = 0  # connection error / timeout
    return code, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", "--requests", type=int, default=500)
    ap.add_argument("-c", "--concurrency", type=int, default=50)
    a = ap.parse_args()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        results = list(ex.map(lambda _: one(a.url), range(a.requests)))
    wall = time.perf_counter() - start

    codes, lat = {}, []
    for code, ms in results:
        codes[code] = codes.get(code, 0) + 1
        lat.append(ms)
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]

    print(f"requests={a.requests} concurrency={a.concurrency} wall={wall:.2f}s")
    print(f"throughput={a.requests / wall:.0f} req/s")
    print(f"latency ms: p50={pct(.50):.1f} p95={pct(.95):.1f} "
          f"p99={pct(.99):.1f} max={lat[-1]:.1f}")
    print(f"status codes: {dict(sorted(codes.items()))}")
    print("(429s = the rate limiter working as intended)")


if __name__ == "__main__":
    main()
