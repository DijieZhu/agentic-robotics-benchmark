"""The members of a site package (the site package format): site.json, state0.json and the five mission/ members
the program reads through ``mission()`` (agent-visible: neutral names only; they carry no real name).

  mission/work_order.txt     WORK_ORDER (the same text for every mission site)
  mission/register.json      the register as the program sees it (``register.public``)
  mission/config_record.json, mission/log.csv, mission/advisory_log.json   the station records (``telemetry_synth``)
"""
from __future__ import annotations

from . import register as RG
from . import sitepack as SP

WORK_ORDER = ("Sortie work order.\n"
              "The station health monitor reported a cabin return-air anomaly during the uncrewed period.\n"
              "Find the cause and restore the return air, stow the loose items you find on the return inlets,\n"
              "keep the station's flight rules and end the sortie on the dock.\n")                       # [C]


def members(site: dict, state0: dict) -> dict:
    """{member name: bytes} of a site package (MANIFEST.json is added by ``sitepack.pack``)."""
    import telemetry_synth as TS
    out = {"site.json": SP.dumps(site), "state0.json": SP.dumps(state0), "mission/work_order.txt": WORK_ORDER.encode("ascii"),
           "mission/register.json": SP.dumps(RG.public(site["register"]))}
    out.update(TS.mission_members(site))
    return out


def package(site: dict, state0: dict) -> bytes:
    return SP.pack(members(site, state0), site["site_id"])


def agent_visible(members_: dict) -> dict:
    return {k: v for k, v in members_.items() if k.startswith("mission/")}
