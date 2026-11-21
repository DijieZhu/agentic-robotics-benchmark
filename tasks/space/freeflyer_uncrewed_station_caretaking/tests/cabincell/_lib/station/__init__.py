"""Station scene v0.3 of the caretaking cell (task-local copy of the station scene; hidden side only).

Built on a vendored copy of the station scene v0.2 (the inlet part family, the site format and its builder, the
robot-free wiring) with the v0.3 additions: the stowage unit (one holding screen and two pouch pads on one screen
blower), its siting rules, the scene data in neutral names (``tests/cabincell/data/station``), the look of the
modules as a task asset (``assets``), map features, zone boxes and per-circuit lighting.

Modules: ``util`` (frames, quaternions), ``data`` (data paths and loaders), ``inlets`` (return-inlet part family),
``stowage`` (stowage unit part, its network and air, datasheet numbers), ``items`` (soft sheets, pouches, the
geometric sealed area), ``scene`` (scene text, look root, near clip, per-circuit headlight), ``sites`` (site
description station_site/2, builder, station part of the validity check), ``site_rig`` (robot-free wiring),
``siting`` (dense ray site check of a stowage unit, air pre-screen), ``mapfeat`` (mapped feature points),
``assets`` (look root lookup and manifest check).
"""
VERSION = "0.3.0"
SITE_SCHEMA = "station_site/2"
