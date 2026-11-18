"""Root-only static verifier of rov_requirements_engineering_from_tender.  Lock down and snapshot before loading the
world or any scientific package.

Separate verification (task.toml: [verifier] environment_mode = "separate"): the engine releases the agent's sandbox,
starts a fresh verifier sandbox from the task image, restores into it only the declared artifact /home/user/submission
(regular files and directories, owned by root; links and special files are never copied) and uploads this folder.
Everything read here is in this folder (grade.py, evaluation_basis.py, schemas.json, grader_config.json, anchor.json,
world/, digest/, the kit), in the image (/opt/venv) or in that restored submission.  The lab daemon does not run in the
verifier and environment/setup/setup.sh does not run there; the agent's own sandbox is gone.

Order (lockdown as in the published requirements tasks):
  1. stop the lab service if one is running (defence in depth for shared verification; alelab_stop, the desk socket);
  2. G1: freeze and kill the agent account's processes, close this folder, snapshot the three submission files into a
     root-only folder (no link followed, at most max_bytes + 1 bytes each) and gate any entry changed after the agent
     phase (ctime / mtime not before the upload of this folder);
  3. load the stage (grade.load_stage): pinned basis and digest, schemas, the world; the truth table, the sets, the
     Part B labels and the delivery bands are recomputed and checked against the world files -- any difference is a
     TASK ERROR, never a zero;
  4. G0 (grade.load_submission: strict JSON, schemas, cross-file rules) and the score (grade.grade); the verdict through
     the kit (reward = min(1, score / full_at)), every metric whitelisted.
The grader executes nothing of the submission: the three files are static JSON.
"""
import json
import os
from pathlib import Path
import pwd
import signal
import stat
import sys
import time

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))
HOME = Path(os.environ.get('ALE_HOME', '/home/user'))
FILES = ('evaluation.json', 'spec_review.json', 'acceptance_schedule.json')
MAX_BYTES = 131072                                   # = grader_config.json submission.max_bytes (checked below)
SNAPSHOT = Path('/opt/uw-req-submission-snapshot')
DESK_SOCKET = '/run/ale-desk/lab.sock'


def live(uid):
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            rows = dict(line.split(':', 1) for line in (p / 'status').read_text().splitlines() if ':' in line)
            if int(rows['Uid'].split()[0]) == uid and not rows['State'].strip().startswith('Z'):
                result.append(int(p.name))
        except (OSError, KeyError, ValueError):
            pass
    return result


def state(pid):
    try:
        for line in Path('/proc/%d/status' % pid).read_text().splitlines():
            if line.startswith('State:'):
                return line.split()[1]
    except (OSError, IndexError):
        pass
    return 'X'


def still_running(uid, bound=3.):
    # SIGSTOP every live process of uid until a pass finds none still running (a respawn that appears meanwhile is
    # stopped by the next pass) or bound elapsed; the PIDs still running (empty: every one is stopped).
    end = time.monotonic() + bound
    while True:
        for pid in live(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        time.sleep(.05)
        running = [pid for pid in live(uid) if state(pid) not in ('T', 't', 'X')]
        if not running or time.monotonic() > end:
            return running


def freeze_and_kill(uid):
    """(processes killed, processes left frozen).  The image's supervisord respawns the account's desktop as fast as
    it is killed, so in the separate verifier a process that outlives the 8 s kill is the image's own: it is stopped
    (a stopped process does not exit, so it is not respawned) and stays stopped while this grader reads the
    submission; only one still running once stopped fails the verify."""
    if uid == 0:
        raise RuntimeError('refusing to kill root')
    end = time.monotonic() + 8
    seen = set()
    frozen = 0
    while pids := live(uid):
        seen.update(pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        again = live(uid)
        if not set(again) - set(pids) or time.monotonic() > end:
            for pid in again:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(.03)
        if time.monotonic() > end:
            if left := live(uid):
                if still_running(uid):
                    raise RuntimeError('agent processes survived lockdown and could not be stopped')
                frozen = len(left)
                print('[grader] %d process(es) of the agent account (the image\'s respawned desktop) outlived the '
                      'kill: stopped' % frozen, file=sys.stderr, flush=True)
            break
    return len(seen), frozen


def stop_lab():
    """Stop the desk's lab service if it runs (it never runs in the separate verifier); True when none is left."""
    try:
        from alelab_stop import stop_all
        rep = stop_all(agent_user=None, socket_path=DESK_SOCKET, strict=False, settle_s=0.0)
        return bool(rep.get('confirmed'))
    except Exception as exc:  # noqa: BLE001 - a failed stop is recorded; the agent kill still runs
        print('[grader] lab stop: %s' % type(exc).__name__, file=sys.stderr, flush=True)
        return False


def snapshot():
    if os.getuid() != 0:
        raise RuntimeError('verify stage must run as root')
    # ctime survives tar mtime preservation; measure before any regular-file chmod.
    start = min(p.lstat().st_ctime for p in STAGE.rglob('*') if p.is_file())
    uid = pwd.getpwnam(HOME.name).pw_uid
    killed, frozen = freeze_and_kill(uid)
    metrics = dict(agent_processes_killed=float(killed), agent_processes_left=float(frozen),
                   gate_g1_submission_integrity=1.)
    for p in STAGE.rglob('*'):
        if not p.is_symlink():
            p.chmod(0o700 if p.is_dir() else 0o600)
    src = HOME / 'submission'
    dst = SNAPSHOT
    dst.mkdir(mode=0o700, exist_ok=True)
    dst.chmod(0o700)
    for p in dst.iterdir():
        p.unlink()
    if not src.exists():
        return dst, metrics
    st = src.lstat()
    if not stat.S_ISDIR(st.st_mode) or max(st.st_ctime, st.st_mtime) >= start:
        metrics['gate_g1_submission_integrity'] = 0.
        return dst, metrics
    for name in FILES:
        p = src / name
        try:
            st = p.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(st.st_mode) or max(st.st_ctime, st.st_mtime) >= start:
            metrics['gate_g1_submission_integrity'] = 0.
            continue
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                metrics['gate_g1_submission_integrity'] = 0.
                continue
            data = f.read(MAX_BYTES + 1)            # an oversized file keeps max + 1 bytes: G0-1 then refuses it
        (dst / name).write_bytes(data)
        (dst / name).chmod(0o600)
    return dst, metrics


def main():
    lab_ok = stop_lab()
    sub, metrics = snapshot()
    metrics['lab_stopped'] = 1. if lab_ok else 0.
    from robotics_grader import fail, read_anchor, ratio_rewards, require, write_verdict
    import grade
    try:
        st = grade.load_stage(STAGE)
    except grade.TaskError as exc:
        fail('stage: %s' % exc)
    require(int(st.cfg['submission']['max_bytes']) == MAX_BYTES and tuple(st.cfg['submission']['files']) == FILES,
            'verify.py and grader_config.json disagree on the submission files')
    value = 0.
    diagnostic = ''
    metrics['gate_g0_structure'] = 0.
    more = grade.zero_metrics()
    if metrics['gate_g1_submission_integrity']:
        docs, problems = grade.load_submission(sub, st)
        if problems:
            diagnostic = 'G0: ' + ' | '.join(problems)
        else:
            try:
                value, more = grade.grade(docs, st)
            except grade.TaskError as exc:
                fail('grade: %s' % exc)
    else:
        diagnostic = 'G1: the submission was linked, special, or changed after the agent phase.'
    metrics.update(more)
    denominator, meta = read_anchor()
    require(meta.get('metric') == st.cfg['reward_key'] and float(meta.get('full_at')) == float(st.cfg['full_at']),
            'anchor.json and grader_config.json disagree on the metric or full_at')
    verdict = ratio_rewards(value, denominator, metric_name=st.cfg['reward_key'], extra=metrics, full_at=meta['full_at'])
    try:
        grade.check_metrics(verdict['metrics'], st.cfg)
    except grade.TaskError as exc:
        fail('metrics: %s' % exc)
    # reward.json and verification.json through the kit and the engine's ale_verify (diagnostic capped as the
    # published requirements tasks cap it).
    write_verdict(verdict, diagnostic=diagnostic[:2000])


if __name__ == '__main__':
    main()
