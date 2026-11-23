# Where the agent account can write in the verifier, and what the grader does about it

"Nothing carries over between sites" is a statement of the instruction. Enumerated in the task image as the agent
account (`find / -writable`, `/proc` and `/sys` left out):

| Writable | What it is | What the grader does |
|---|---|---|
| `/usr/local/include`, `/usr/local/lib`, `/usr/local/share`, `/usr/local/bin` (about 1,040 directories) | trees the base image gives to the account | once, before any agent code: every file and directory the account owns becomes root's (`close_agent_writable`) |
| `/home/user` and what is under it (22 directories) | the account's home; the submission is restored here | the same; the submission was copied to a read-only place first |
| `/tmp/runtime-user` | the desktop's runtime directory | the same |
| `/tmp`, `/var/tmp`, `/run/lock`, `/dev/shm`, `/dev/mqueue` | world-writable directories | before every site: every entry the account owns is removed (`purge_agent_files`) |
| world-writable files of other owners | none in this image; the walk takes the world-write bit off any it finds | once |
| the program's own home `/opt/arb-run/home-<site>` | made by the grader for one site | removed before the next site |
| System V shared memory, semaphores, message queues | not files | before every site: `ipcrm --all` as the account |

Processes of the account are killed after every site. Checked with a probe submission that writes into each of these
places and lists what it finds on the next sites: after the close the account can write only in the five
world-writable directories and its own per-site home, and on every later site it finds nothing of an earlier one.
