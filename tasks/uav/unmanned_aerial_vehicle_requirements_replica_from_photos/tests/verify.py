"""Root-only static verifier. Lock down and snapshot before loading truth or scientific packages.

Separate verification (task.toml: [verifier] environment_mode = "separate"). The engine releases the agent's sandbox,
starts a fresh verifier sandbox from the task image and restores into it only the declared artifact
/home/user/submission (regular files and directories, owned by root; links and special files are never copied), then
uploads this folder. Everything read here is in this folder (gt.json, schemas.json, anchor.json, grade.py,
procedure.py, the kit), in the image (/opt/venv) or in that restored submission. environment/setup/setup.sh
does not run in the verifier, so the givens it stages into the agent home are not there, and none of the agent's files
besides the submission is read. given_hashes.json stays in this folder as the maintainers' pin of the givens; this
grader does not read it. In the separate verifier nothing the agent started is running: the freeze and kill stop only
the image's own desktop session, which the base image's supervisord runs as the agent account and which this grader
does not use. The engine restores the submission before it uploads this folder, so no restored entry is late; the
lockdown, the snapshot and gate G1 stay as defence in depth. A submission root that is not a directory still fails G1;
a symbolic-link root is not restored at all (the engine copies no link), so G0 then finds the three files missing.
"""
import json
import os
from pathlib import Path
import pwd
import signal
import stat
import sys
import time

STAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(STAGE))
HOME=Path(os.environ.get('ALE_HOME','/home/user'))
FILES=('requirements.yaml','bom.json','kinematic_sketch.json')


def live(uid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            rows=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
            if int(rows['Uid'].split()[0])==uid and not rows['State'].strip().startswith('Z'):result.append(int(p.name))
        except (OSError,KeyError,ValueError):pass
    return result


def freeze_and_kill(uid):
    if uid==0:raise RuntimeError('refusing to kill root')
    end=time.monotonic()+8
    seen=set()
    while pids:=live(uid):
        seen.update(pids)
        for pid in pids:
            try:os.kill(pid,signal.SIGSTOP)
            except ProcessLookupError:pass
        again=live(uid)
        if not set(again)-set(pids) or time.monotonic()>end:
            for pid in again:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        time.sleep(.03)
        if time.monotonic()>end:
            if live(uid):raise RuntimeError('agent processes survived lockdown')
            break
    return len(seen)


def snapshot():
    if os.getuid()!=0:raise RuntimeError('verify stage must run as root')
    # ctime survives tar mtime preservation; measure before any regular-file chmod.
    start=min(p.lstat().st_ctime for p in STAGE.rglob('*') if p.is_file())
    uid=pwd.getpwnam(HOME.name).pw_uid
    killed=freeze_and_kill(uid)
    metrics=dict(agent_processes_killed=float(killed),agent_processes_left=0.,gate_g1_submission_integrity=1.)
    for p in STAGE.rglob('*'):
        if not p.is_symlink():p.chmod(0o700 if p.is_dir() else 0o600)
    src=HOME/'submission';dst=Path('/opt/uav-submission-snapshot')
    dst.mkdir(mode=0o700,exist_ok=True);dst.chmod(0o700)
    for p in dst.iterdir():p.unlink()
    if not src.exists():return dst,metrics
    st=src.lstat()
    if not stat.S_ISDIR(st.st_mode) or max(st.st_ctime,st.st_mtime)>=start:
        metrics['gate_g1_submission_integrity']=0.;return dst,metrics
    for name in FILES:
        p=src/name
        try:st=p.lstat()
        except FileNotFoundError:continue
        if not stat.S_ISREG(st.st_mode) or st.st_size>262144 or max(st.st_ctime,st.st_mtime)>=start:
            metrics['gate_g1_submission_integrity']=0.;continue
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as f: data=f.read(262145)
        if len(data)>262144:metrics['gate_g1_submission_integrity']=0.;continue
        (dst/name).write_bytes(data);(dst/name).chmod(0o600)
    return dst,metrics


def main():
    sub,metrics=snapshot()
    from robotics_grader import read_anchor,ratio_rewards,write_verdict
    import grade
    truth=json.loads((STAGE/'gt.json').read_text())
    schemas=json.loads((STAGE/'schemas.json').read_text())
    value=0.;diagnostic=''
    metrics['gate_g0_structure']=0.
    if metrics['gate_g1_submission_integrity']:
        try:docs=grade.load_submission(sub,schemas)
        except (OSError,ValueError,TypeError,RecursionError,grade.yaml.YAMLError,grade.jsonschema.ValidationError) as exc:
            diagnostic='G0: '+str(exc)[:1600]
        else:value,more=grade.grade(docs,truth);metrics.update(more)
    else:diagnostic='G1: submission was linked, special, oversized, or changed after the agent phase.'
    denominator,meta=read_anchor()
    verdict=ratio_rewards(value,denominator,metric_name='requirements_score',extra=metrics,full_at=meta['full_at'])
    # reward.json and verification.json through the kit and the engine's ale_verify. diagnostic[:2000] is the earlier
    # ale_verify API's diagnostic cap, kept so the recorded text is unchanged; every metric is recorded as float(v) and
    # a non-finite one still fails the verdict, as rec.stat did.
    write_verdict(verdict,diagnostic=diagnostic[:2000])


if __name__=='__main__':main()
