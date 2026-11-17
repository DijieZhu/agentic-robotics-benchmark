"""The leak audit's spy (a maintainer's tool, never part of a submission an agent sees): run.sh starts it in the
background as the agent's account, under the same netguard as the supervisor, beside the reference expert, for one
full verify. It records what of the grader, the simulator and the network it can reach, as JSON lines in spy.jsonl
next to itself (the submission folder; regrade.sh keeps that folder as the supervisor left it).

Tries: internet sockets (AF_INET and AF_INET6, TCP and UDP, to the internet and to the loopback, every listening
loopback port included: CARLA's RPC on 2000 and the Traffic Manager on 2050), a name lookup, AF_PACKET, AF_NETLINK,
AF_VSOCK and a raw socket, an AF_UNIX pair (the control: it must work); the grader's folder (/opt/ale/tests: the set,
the selection, the stagings, the config, the anchor, the token digest); the bridge's copy in /run (drive-bridge-*:
drive_variants.py, drive_bridge.py), the guard's folder (drive-guard-*), the simulator's log (/tmp/drive-simlog-*);
/proc/<pid>/cmdline, environ, maps, mem and fd of every process of another account (root's grader, the bridge and
CARLA as the simulator account); /proc/sys/kernel/yama/ptrace_scope. Every 10 s while the verify runs."""
import glob
import json
import os
import socket
import time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spy.jsonl")
ME = os.getuid()


def rec(kind, **kw):
    with open(OUT, "a") as f:
        f.write(json.dumps(dict(t=round(time.time(), 1), pid=os.getpid(), kind=kind, **kw)) + "\n")


def attempt(fn):
    try:
        got = fn()
        if isinstance(got, (bytes, str)):
            got = (got.decode(errors="replace") if isinstance(got, bytes) else got)[:400]
        elif not isinstance(got, (int, float, list, dict, type(None))):
            got = repr(got)[:400]
        return dict(ok=True, got=got)
    except BaseException as exc:  # noqa: BLE001
        return dict(ok=False, err="{}: {}".format(type(exc).__name__, exc)[:200])


def read(path, n=4096):
    with open(path, "rb") as f:
        return f.read(n).decode(errors="replace")


def listen_ports():
    out = []
    for fam in ("tcp", "tcp6"):
        try:
            lines = open("/proc/net/{}".format(fam)).read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            if len(f) > 3 and f[3] == "0A":
                out.append((fam, int(f[1].split(":")[1], 16)))
    return sorted(set(out))


def tcp(fam, host, port):
    def go():
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.settimeout(3)
        try:
            s.connect((host, port))
            s.sendall(b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
            return s.recv(200)
        finally:
            s.close()
    return go


def udp(fam, host, port):
    def go():
        s = socket.socket(fam, socket.SOCK_DGRAM)
        s.settimeout(3)
        try:
            s.sendto(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01", (host, port))
            return s.recvfrom(512)[0].hex()[:80]
        finally:
            s.close()
    return go


def raw(fam, typ, proto=0):
    def go():
        s = socket.socket(fam, typ, proto)
        s.close()
        return "opened"
    return go


def unix_pair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    a.sendall(b"ping")
    got = b.recv(4).decode()
    a.close()
    b.close()
    return got


def net():
    tries = {
        "inet tcp 1.1.1.1:443": tcp(socket.AF_INET, "1.1.1.1", 443),
        "inet tcp 127.0.0.1:2000 (CARLA RPC)": tcp(socket.AF_INET, "127.0.0.1", 2000),
        "inet tcp 127.0.0.1:2050 (Traffic Manager)": tcp(socket.AF_INET, "127.0.0.1", 2050),
        "inet udp 8.8.8.8:53": udp(socket.AF_INET, "8.8.8.8", 53),
        "inet udp 127.0.0.1:53": udp(socket.AF_INET, "127.0.0.1", 53),
        "inet6 tcp [2606:4700:4700::1111]:443": tcp(socket.AF_INET6, "2606:4700:4700::1111", 443),
        "inet6 tcp [::1]:2000": tcp(socket.AF_INET6, "::1", 2000),
        "inet6 udp [::1]:53": udp(socket.AF_INET6, "::1", 53),
        "getaddrinfo example.com": lambda: [a[4] for a in socket.getaddrinfo("example.com", 443)][:3],
        "inet raw icmp": raw(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP),
        "inet socketpair": lambda: socket.socketpair(socket.AF_INET, socket.SOCK_STREAM),
        "packet": raw(socket.AF_PACKET, socket.SOCK_RAW),
        "netlink route": raw(socket.AF_NETLINK, socket.SOCK_RAW, 0),
        "vsock": raw(40, socket.SOCK_STREAM),
        "unix socketpair (control: must work)": unix_pair,
    }
    for fam, port in listen_ports():
        if fam == "tcp":
            tries["inet tcp 127.0.0.1:{} (listening)".format(port)] = tcp(socket.AF_INET, "127.0.0.1", port)
        else:
            tries["inet6 tcp [::1]:{} (listening)".format(port)] = tcp(socket.AF_INET6, "::1", port)
    rec("net", listening=listen_ports(), results={k: attempt(fn) for k, fn in tries.items()})


def files():
    res = {}
    for p in ("/opt/ale/tests/hidden/episodes.json", "/opt/ale/tests/hidden/selection.json",
              "/opt/ale/tests/drive_variants.py", "/opt/ale/tests/verify.py", "/opt/ale/tests/grader_config.json",
              "/opt/ale/tests/anchor.json", "/opt/ale/tests/oracle_token.sha256"):
        res[p] = attempt(lambda p=p: read(p, 400))
    for d in ("/opt/ale", "/opt/ale/tests", "/opt/ale/tests/hidden", "/root", "/run", "/tmp"):
        res["listdir " + d] = attempt(lambda d=d: sorted(os.listdir(d)))
    for d in sorted(glob.glob("/run/drive-bridge-*")) + sorted(glob.glob("/run/drive-guard-*")) + \
            sorted(glob.glob("/tmp/drive-simlog-*")):
        res["listdir " + d] = attempt(lambda d=d: sorted(os.listdir(d)))
        for name in ("drive_variants.py", "drive_bridge.py", "probe_sprites", "netguard.py", "run.sh", "sim.log"):
            p = os.path.join(d, name)
            res[p] = attempt(lambda p=p: sorted(os.listdir(p)) if os.path.isdir(p) else read(p, 400))
    res["ptrace_scope"] = attempt(lambda: read("/proc/sys/kernel/yama/ptrace_scope").strip())
    rec("files", results=res)


def procs(seen):
    for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        try:
            st = os.stat("/proc/{}".format(pid))
        except OSError:
            continue
        if st.st_uid == ME or pid in seen:
            continue
        seen.add(pid)
        out = dict(target=pid, uid=st.st_uid)
        for what in ("cmdline", "environ", "maps", "mem"):
            r = attempt(lambda what=what: read("/proc/{}/{}".format(pid, what), 65536 if what != "mem" else 16))
            if r["ok"] and what in ("cmdline", "environ"):
                r["got"] = r["got"].replace("\0", " ")
            out[what] = r
        out["fd"] = attempt(lambda: sorted(os.listdir("/proc/{}/fd".format(pid))))
        rec("proc", **out)


def main():
    rec("start", uid=ME, gid=os.getgid(), groups=os.getgroups(), cwd=os.getcwd(), env=dict(os.environ))
    seen = set()
    k = 0
    while True:
        for fn in ((lambda: procs(seen)),) + ((net, files) if k % 3 == 0 else ()):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                rec("error", err=repr(exc)[:300])
        k += 1
        time.sleep(10)


if __name__ == "__main__":
    main()
