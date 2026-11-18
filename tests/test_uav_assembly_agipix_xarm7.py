"""uav_assembly_agipix_xarm7: the network audit of the submitted program (tests/netaudit.py) and how
tests/verify.py applies it.

The program runs during verify, after the engine has opened the network. The static scan flags each forbidden form in
the submission's own .py files, parsed as the interpreter parses them (a coding declaration included, every folder
included); honest code and the shipped client, runner and oracle program scan clean; the namespace counters see a
connection shorter than any sampling period; the per-account socket count reads the owner-uid column of /proc/net (in
the engine's sandbox root has no CAP_SYS_PTRACE, so another account's /proc/<pid>/fd links are unreadable); verify.py
turns a static hit into a zero verdict with a reason and a metric, and an episode's network record into a reason only
when the audit is enforced and something was seen.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1] / "tasks/uav/uav_assembly_agipix_xarm7"
_NAMES = ("netaudit", "verify", "verify_freeze")


def _load():
    """The task's netaudit and verify modules, without a __pycache__ in the task folder and without leaving the names
    `verify` / `netaudit` / `verify_freeze` behind for other tests."""
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    saved_modules = {name: sys.modules.pop(name) for name in _NAMES if name in sys.modules}
    sys.dont_write_bytecode = True
    try:
        sys.path.insert(0, str(TASK / "tests"))
        import netaudit
        import verify
        return netaudit, verify
    finally:
        for name in _NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag


N, V = _load()

FORBIDDEN = {
    "a.py": "import importlib\nm = importlib.import_module('so' + 'cket')\n",
    "b.py": "from multiprocessing.connection import Client\n",
    "c.py": "import logging.handlers\n",
    "d.py": "import subprocess\nsubprocess.run(['true'])\n",
    "e.py": "exec(open('x').read())\n",
    "f.py": "import socket\n",
    "g.py": "import os\nos.system('curl x')\n",
    "h.py": "import ctypes\n",
    "i.py": "from urllib.request import urlopen\n",
    "j.py": "name = 'x'\nm = __import__(name)\n",
    "k.py": "from os import execv\nexecv('/bin/sh', ['sh'])\n",
    "l.py": "import http.client\n",
    "m.py": "import asyncio\n",
    "pkg/n.py": "import requests\n",
    "__pycache__/o.py": "import ssl\n",
}
# `import socket` hidden in a comment for a UTF-8 reader; the interpreter decodes +AAo- as a line break
UTF7 = b"# coding: utf-7\n# a comment +AAo-import socket as s+AAo-print(s)\n"


def _write(root, files):
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(src if isinstance(src, bytes) else src.encode())


def test_static_scan_flags_each_forbidden_form(tmp_path):
    _write(tmp_path, dict(FORBIDDEN, **{"utf7.py": UTF7}))
    hits = N.scan_imports(str(tmp_path))
    assert {rel for rel, _ in hits} == set(FORBIDDEN) | {"utf7.py"}
    assert ("utf7.py", "socket") in hits


def test_static_scan_passes_honest_code(tmp_path):
    _write(tmp_path, {
        "assemble.py": ("import numpy as np\nimport scipy.optimize\nimport mujoco\nimport trimesh\nfrom PIL import Image\n"
                        "import json, os, math, multiprocessing\nfrom concurrent.futures import ThreadPoolExecutor\n"
                        "import importlib\nhelpers = importlib.import_module('helpers')\nx = eval('1 + 2')\n"
                        "def solve(cell):\n    return cell.get_state()\n"),
        "helpers.py": "from pathlib import Path\nHERE = Path(__file__).parent\n",
        "notes.txt": "import socket\n",                        # not Python source
        "broken.py": "def (:\n",                              # the interpreter cannot import it either
    })
    assert N.scan_imports(str(tmp_path)) == []


def test_the_shipped_programs_scan_clean(tmp_path):
    assert N.scan_imports(str(TASK / "environment/setup/payload")) == []
    assert N.scan_imports(str(TASK / "solution")) == []      # the oracle's program
    for name in ("cell_client.py", "cell_runner.py"):
        (tmp_path / name).write_bytes((TASK / "tests/setup_payload" / name).read_bytes())
    (tmp_path / "assemble.py").write_bytes((TASK / "solution/assemble.py").read_bytes())
    assert V.static_network_audit(tmp_path) == []


@pytest.mark.skipif(not os.path.isfile("/proc/net/snmp"), reason="Linux /proc/net/snmp needed")
def test_counters_see_a_connection_shorter_than_any_sample():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    with N.NetCounters() as counters:
        client = socket.create_connection(server.getsockname(), timeout=2.0)
        client.close()
    server.close()
    assert counters.delta["tcp_active_opens"] >= 1 and counters.total >= 1


TABLE_HEAD = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"


def _row(i, local, remote, uid, inode):
    return "%4d: %s %s 01 00000000:00000000 00:00000000 00000000 %5d        0 %d 1 0000000000000000 20 4 30 10 -1\n" % (
        i, local, remote, uid, inode)


def test_remote_sockets_by_the_owner_uid(tmp_path):
    tables = {
        "tcp": [_row(0, "0100007F:1F90", "00000000:0000", 65534, 101),      # listening: unspecified remote
                _row(1, "0200000A:D2F0", "0100000A:01BB", 65534, 102),      # 10.0.0.1:443: remote
                _row(2, "0100007F:D2F1", "0100007F:1F90", 65534, 103),      # loopback
                _row(3, "0200000A:D2F2", "0100000A:01BB", 1000, 104)],      # another account
        "tcp6": [_row(0, "0" * 32 + ":1F90", "00000000000000000000000001000000:D2F3", 65534, 105),   # ::1
                 _row(1, "0" * 32 + ":1F91", "0000000000000000FFFF00000100007F:D2F4", 65534, 106),   # ::ffff:127.0.0.1
                 _row(2, "0" * 32 + ":1F92", "B80D0120000000000000000001000000:01BB", 65534, 107)],  # 2001:db8::1
        "udp": [_row(0, "0200000A:A000", "08080808:0035", 65534, 108)],     # a connected UDP socket, 8.8.8.8:53
        "udp6": [],
    }
    for name, rows in tables.items():
        (tmp_path / name).write_text(TABLE_HEAD + "".join(rows))
    assert N.remote_sockets(65534, (), proc_net=str(tmp_path)) == 3
    assert N.remote_sockets(1000, (), proc_net=str(tmp_path)) == 1
    assert N.remote_sockets(4242, (), proc_net=str(tmp_path)) == 0
    assert N.remote_sockets(None, (), proc_net=str(tmp_path)) == 0


@pytest.mark.skipif(not os.path.isfile("/proc/net/udp"), reason="Linux /proc/net needed")
def test_the_sampler_counts_a_held_remote_socket():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))          # TEST-NET-1; a UDP connect sends nothing, it only fixes the remote
    except OSError:
        probe.close()
        pytest.skip("no route for a remote address here")
    try:
        assert N.remote_sockets(os.getuid(), ()) >= 1             # by the owner uid
        assert N.remote_sockets(None, [os.getpid()]) >= 1         # by the held inode (one's own process)
        sampler = N.SocketSampler(uid=os.getuid(), period_s=0.05)
        deadline = time.monotonic() + 10
        with sampler:
            while sampler.samples < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
        assert sampler.max_seen >= 1
    finally:
        probe.close()


def test_verify_reasons():
    assert "assemble.py: socket" in V.static_network_reason([("assemble.py", "socket")])
    assert "and 2 more" in V.static_network_reason([("a.py", "socket")] * 7)
    quiet = dict(enforced=True, outbound=0, delta={"tcp_active_opens": 0}, remote_sockets_seen=0)
    assert V.runtime_network_reason(0, quiet) is None
    loud = dict(enforced=True, outbound=2, delta={"tcp_active_opens": 1, "udp_out": 1}, remote_sockets_seen=0)
    reason = V.runtime_network_reason(1, loud)
    assert "episode 1" in reason and "tcp_active_opens +1" in reason and "udp_out +1" in reason
    held = dict(enforced=True, outbound=0, delta={}, remote_sockets_seen=1)
    assert "remote sockets of the runner account seen: 1" in V.runtime_network_reason(2, held)
    assert V.runtime_network_reason(0, dict(loud, enforced=False)) is None   # a host run records, never voids


def test_verify_zeroes_a_static_hit(tmp_path, monkeypatch):
    home = tmp_path / "agipix-netaudit-test-home"
    _write(home / "submission", {"assemble.py": "import socket\n\ndef solve(cell):\n    return None\n"})
    verdict_path = tmp_path / "verdict.json"
    monkeypatch.setattr(V, "TASK_HOME", home)
    monkeypatch.setattr(V, "root_assets_check", lambda: None)   # the image's /opt/arb is not on this host
    monkeypatch.setenv("ALE_STAGE_DIR", str(TASK / "tests"))
    monkeypatch.setenv("ROBOTICS_GRADER_VERDICT_PATH", str(verdict_path))
    monkeypatch.delenv("ALE_VERIFY_PYTHON", raising=False)
    V.main()
    verdict = json.loads(verdict_path.read_text())
    assert verdict["rewards"]["reward"] == 0.0
    assert verdict["metrics"]["network_imports"] == 1.0 and verdict["metrics"]["network_audit_ok"] == 0.0
    assert "network audit" in verdict["diagnostic"] and "assemble.py: socket" in verdict["diagnostic"]
