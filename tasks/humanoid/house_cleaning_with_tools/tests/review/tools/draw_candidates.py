"""Draw the hidden-set candidates: N 63-bit seeds from os.urandom, in drawn order. draw_candidates.py N OUT.json"""
import json, os, sys, time
n, out = int(sys.argv[1]), sys.argv[2]
seeds = [int.from_bytes(os.urandom(8), "big") & ((1 << 63) - 1) for _ in range(n)]
json.dump({"drawn_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "os.urandom(8) & (2^63 - 1)",
           "candidates": seeds}, open(out, "w"), indent=1)
print(seeds)
