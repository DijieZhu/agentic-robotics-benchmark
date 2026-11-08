"""Download the pinned subset of NVIDIA's public Isaac Sim asset library at image build and check every file.

    python3 fetch_isaac_assets.py isaac_assets.sha256 /opt/assets/isaac

The list starts with `# bucket:` and `# root:` header lines, then one `<sha256>  <size>  <path>` line per file; a file
lands at `<dest>/<path>`, so the library's own relative references keep working. Standard library only.
"""

import concurrent.futures
import hashlib
import os
import sys
import urllib.parse
import urllib.request


def fetch(bucket, root, dest, sha, size, rel):
    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{bucket}/{urllib.parse.quote(root + rel)}"
    for attempt in range(4):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=300) as r, open(out + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != sha or os.path.getsize(out + ".part") != size:
                raise ValueError(f"checksum mismatch for {rel}")
            os.replace(out + ".part", out)
            return
        except Exception:
            if attempt == 3:
                raise


def main():
    listing, dest = sys.argv[1], sys.argv[2]
    head, rows = {}, []
    for line in open(listing):
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            head[key.strip()] = value.strip()
        elif line.strip():
            sha, size, rel = line.split(None, 2)
            rows.append((sha, int(size), rel.strip()))
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda r: fetch(head["bucket"], head["root"], dest, *r), rows))
    print(f"[image] {len(rows)} Isaac asset files, {sum(r[1] for r in rows) / 1e6:.0f} MB in {dest}")


if __name__ == "__main__":
    main()
