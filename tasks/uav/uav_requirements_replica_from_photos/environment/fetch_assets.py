"""Fetch a task's large agent-visible files from the benchmark's asset repository at image build time.

    python3 fetch_assets.py <assets.sha256> <dest_dir>

The canonical copy is `shared/task_assets/fetch_assets.py` in the benchmark repository. Every task that has assets
carries a byte-identical copy in its `environment/` (the image's build context); `scripts/task_assets.py check`
enforces the identity.

`assets.sha256` (in environment/, so in the build context) holds a header, then one `<sha256>  <relative path>` line per
file, as `sha256sum` writes them:

    # endpoint: https://huggingface.co
    # repo: grounded-intelligence-initiative/ale-robotics-assets
    # revision: <40-hex commit of the dataset repository>
    # prefix: tasks/<platform>/<task name>/payload
    <sha256>  photos/photo_01.jpg

`endpoint` and `repo` are written from the repository's `task-assets.toml` by `scripts/task_assets.py sync`: change
that file, not these lines (an absent `endpoint` means https://huggingface.co). Every file is downloaded from
`<endpoint>/datasets/<repo>/resolve/<revision>/<prefix>/<relative path>` and written to `<dest_dir>/<relative path>`
only if its sha256 matches; any mismatch or failed download fails the build. Files are downloaded `WORKERS` at a
time: each request waits on the Hub's redirect far longer than it transfers, so one at a time a list of hundreds of
small files takes minutes. Stdlib only (runs before any venv).
"""
import concurrent.futures
import hashlib
import os
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_ENDPOINT = "https://huggingface.co"
URL = "{endpoint}/datasets/{repo}/resolve/{rev}/{path}"
WORKERS = 16
CHUNK = 1 << 20


def read_list(path):
    head, files = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                head[k.strip()] = v.strip()
                continue
            digest, rel = line.split(None, 1)
            files.append((digest.lower(), rel.strip().lstrip("*")))
    for k in ("repo", "revision", "prefix"):
        if not head.get(k):
            sys.exit("assets list %s has no '# %s:' header" % (path, k))
    return head, files


def download(url, out, tries=4):
    """Stream `url` into `out`, retrying a failed attempt; returns the sha256 of what was written."""
    for k in range(tries):
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as fh:
                for chunk in iter(lambda: r.read(CHUNK), b""):
                    digest.update(chunk)
                    fh.write(chunk)
            return digest.hexdigest()
        except Exception as exc:  # noqa: BLE001  (retried, then fatal)
            if k == tries - 1:
                raise RuntimeError("download failed: %s (%s)" % (url, exc)) from None
            time.sleep(5 * (k + 1))


def fetch(url, digest, rel, dest):
    out = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    got = download(url, out + ".part")
    if got != digest:
        raise RuntimeError("sha256 mismatch for %s: %s != %s" % (rel, got, digest))
    os.replace(out + ".part", out)


def main():
    lst, dest = sys.argv[1], sys.argv[2]
    head, files = read_list(lst)
    endpoint = (head.get("endpoint") or DEFAULT_ENDPOINT).rstrip("/")
    for _, rel in files:
        if os.path.isabs(rel) or ".." in rel.split("/"):
            sys.exit("unsafe path in assets list: %s" % rel)
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        jobs = [
            pool.submit(fetch, URL.format(endpoint=endpoint, repo=head["repo"], rev=head["revision"],
                                          path=urllib.parse.quote(head["prefix"] + "/" + rel)), digest, rel, dest)
            for digest, rel in files
        ]
        for job in concurrent.futures.as_completed(jobs):
            try:
                job.result()
            except Exception as exc:  # noqa: BLE001  (the first failure fails the build)
                pool.shutdown(cancel_futures=True)
                sys.exit(str(exc))
    print("[assets] %d file(s) from %s/%s@%s verified into %s" % (len(files), endpoint, head["repo"],
                                                                 head["revision"][:12], dest))


if __name__ == "__main__":
    main()
