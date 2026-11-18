"""ttverify: the race-day grader's own pieces around the vendored cell (grader side; the agent never sees them).

    netkill.py    the second seccomp filter every process of a program runs under: a socket of any family but
                  AF_UNIX, io_uring and a foreign syscall ABI KILL the process (SIGSYS), so a network attempt is
                  recorded by the program's exit status (the house filter `cellkit.nonet`, installed first, only
                  refuses it); a new process is refused (fork, vfork, clone without CLONE_THREAD: EPERM; clone3:
                  ENOSYS, so threads fall back to clone); stdlib only
    regime.py     how a program is run: the account switch (setpriv: the program keeps the pid the cell's CPU
                  accounting reads), the limits (prlimit: CPU backstop, address space, file size, no core), the two
                  filters, the per-run directory, the import check, the survivor check and the sweep, the self-tests;
                  stdlib only
    hiddenset.py  the hidden set: tests/hidden/selection.json, the venues' world files and reference records, their
                  pinned sha256 and layout digests, the oracle token's sha256
    manifest.py   the vendored files' sha256 (tests/grader_manifest.json), checked before anything of them is imported
"""
