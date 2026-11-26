"""The interface through which a world hook takes part in a unit. PUBLIC.

PUBLIC: this module holds no hook of its own and imports nothing but the standard library. `unit.py` imports it and
never imports a world module: the empty `HookSet` is the default world. A world (rests, a weight, an object, grasps, a
push, a point mass at a tool site) implements `Hook` and hands a `HookSet` to `OpenArmUnit.build` or to the public
runner of a submitted model.

What a hook may do, and when (in the order of one physics step of the unit):

    declare(spec)                 before compilation: add its bodies, geoms, sites and equalities to the MjSpec (every
                                  body a hook uses exists in the compiled model)
    bind(unit)                    after compilation, with the unit at its default pose, disabled: resolve its ids in
                                  unit.model; read nothing that varies
    per_step_fields()             after bind, asked once: the names of the per-step fields it offers (a tuple; default
                                  none)
    reset(unit)                   at the start of OpenArmUnit.reset (before the pose is set)
    set_world_state(unit, state)  from OpenArmUnit.set_state (object poses, active welds ...)
    pre_step(unit, n)             before physics step n: forces ADDED to data.qfrc_applied / data.xfrc_applied (the
                                  unit zeroes both at the step's start where the bus layer does not write them; the
                                  bus layer ASSIGNS qfrc_applied at the sixteen driven dofs before the hooks run), then
                                  the switches due at step n (welds, stops of held necks, ramps)
    post_step(unit, n)            after mj_step and the bus layer's update: event detection
    per_step(unit, n, names)      after the post_step of every hook, at every step at which a trace profile asks for
                                  some of its fields: {name: value} for each of `names` (copies; cheap: values the
                                  hook already holds or reads from the state, no solver call)
    events()                      its events since the last call (dicts with 'kind', 'step' and fields; drained)
    truth(unit)                   its part of the truth: a JSON-able mapping (copies)
    state() / load_state(s)       everything it changes at run time, as JSON-able data (floats exact); a snapshot
                                  holds it and the state digest covers it (so it holds no buffer a caller drains)
    model_fields(unit) /          every MODEL field it changes at run time (eq_data of a weld being ramped, the finger
    load_model_fields(unit, f)    ranges of a held neck, a per-job load ...); a snapshot holds them
    declaration()                 what was declared (JSON-able): the world digest is the sha256 of the HookSet's
                                  canonical declaration (HookSet.digest)
    contact_classes()             {body or geom name: class name} for the contact classes of its bodies and parts

What a hook may read, and when. MuJoCo's derived arrays in unit.data (body, site and geom poses: xpos, xquat, xmat,
site_xpos, site_xmat, geom_xpos, geom_xmat ...; constraint and contact arrays: efc_*, contact, ncon, nefc ...) belong to
the computation that last wrote them, and that is not always the last physics step: the unit's truth call recomputes
body and site poses at the present state, a snapshot does so for the world's truth, and a restore recomputes every
derived array at the restored state. A hook therefore:
    - in pre_step and set_world_state reads a derived POSITION array only after computing it itself in that same call
      (mujoco.mj_kinematics(unit.model, unit.data) on the present state, or mj_forward where it needs more), and
      never reads constraint or contact arrays there;
    - reads constraint and contact arrays only in post_step (they belong to the step just solved) and in truth (the
      unit serves the world's truth from the values a snapshot stored, right after a restore);
    - reads the integration state (qpos, qvel, act, time) and its own state at any time.
A world that follows the rule gives runs whose motion and state digests do not depend on whether, when or how often
the truth is read or a snapshot is taken between the exchanges; a world that breaks it does not (the unit cannot
detect it; comparing the digests of runs with and without such calls does).

A trace profile names a hook's field as "<hook name>.<field>"; an entry that no hook offers is refused when the unit is
built. A hook uses integers for every schedule (no floating-point clock) and, on the per-step path, float64 element-wise
arithmetic only (no BLAS, no reduction of floats, no transcendental function, no wall clock). Every method has a default
that does nothing, so that a hook implements only what it needs.
"""
import hashlib
import json

HOOKS_FORMAT = "openarm_unit/HookSet/1"


class HookError(ValueError):
    """A hook set that cannot be used (two hooks of one name, a hook state of another set, an unknown per-step field)."""


class Hook:
    """The base of a world hook (see the module text). `name` identifies the hook inside its HookSet."""

    name = "hook"

    def declare(self, spec):
        return None

    def bind(self, unit):
        return None

    def per_step_fields(self):
        return ()

    def reset(self, unit):
        return None

    def set_world_state(self, unit, state):
        """A derived position array is read here only after this call computed it (mj_kinematics); no constraint or
        contact array is read here (module text)."""
        return None

    def pre_step(self, unit, n):
        """A derived position array is read here only after this call computed it (mj_kinematics on the present
        state); no constraint or contact array is read here (module text)."""
        return None

    def post_step(self, unit, n):
        """Constraint and contact arrays read here belong to the step just solved (module text)."""
        return None

    def per_step(self, unit, n, names):
        return {}

    def events(self):
        return []

    def truth(self, unit):
        """Called with body and site poses computed at the present state; right after a restore the unit serves the
        value a snapshot stored instead of calling this (module text)."""
        return {}

    def state(self):
        return {}

    def load_state(self, s):
        return None

    def model_fields(self, unit):
        return {}

    def load_model_fields(self, unit, fields):
        return None

    def declaration(self):
        return {"kind": type(self).__name__}

    def contact_classes(self):
        return {}


class HookSet:
    """An ordered set of hooks (the world of a unit). `HookSet()` is the default world: no hook, nothing declared."""

    def __init__(self, hooks=()):
        self.hooks = tuple(hooks)
        names = [h.name for h in self.hooks]
        if len(set(names)) != len(names):
            raise HookError(f"two hooks share a name: {names}")
        for h in self.hooks:
            if not isinstance(h, Hook):
                raise HookError(f"{h!r} is not a Hook")

    def __bool__(self):
        return bool(self.hooks)

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    # -------------------------------------------------------------------------------------------- build time
    def declare(self, spec):
        for h in self.hooks:
            h.declare(spec)

    def bind(self, unit):
        for h in self.hooks:
            h.bind(unit)

    def declaration(self):
        return {"format": HOOKS_FORMAT, "hooks": [{"name": h.name, "declaration": h.declaration()}
                                                  for h in self.hooks]}

    def digest(self):
        """sha256 of the canonical declaration (sorted keys, floats by repr()): the world digest of a snapshot."""
        text = json.dumps(self.declaration(), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def contact_classes(self):
        out = {}
        for h in self.hooks:
            out.update(h.contact_classes())
        return out

    # ------------------------------------------------------------------------------------------ per-step fields
    def per_step_fields(self):
        """Every per-step field the hooks offer, as trace-profile entries "<hook name>.<field>", in hook order."""
        return tuple(self._offered())

    def offered_fields(self):
        """{"<hook name>.<field>": (hook, field)} of every per-step field the hooks offer, in hook order: each hook's
        per_step_fields() asked once (a unit asks this once, after bind, and makes its trace plan from it)."""
        return self._offered()

    def _offered(self):
        out = {}
        for h in self.hooks:
            for f in tuple(h.per_step_fields()):
                key = f"{h.name}.{f}"
                if key in out:
                    raise HookError(f"two hooks offer the per-step entry {key!r}")
                out[key] = (h, str(f))
        return out

    def per_step_plan(self, entries, offered=None):
        """The plan of the per-step calls for trace-profile entries "<hook name>.<field>": a tuple of (hook, names,
        entries) in hook order. HookError for an entry no hook offers. offered: what offered_fields() returned (the
        hooks are not asked again), or None (they are asked)."""
        entries = tuple(entries)
        offered = self._offered() if offered is None else dict(offered)
        bad = [e for e in entries if e not in offered]
        if bad:
            raise HookError(f"per-step entries {bad} are offered by no hook (offered: {sorted(offered)})")
        plan = []
        for h in self.hooks:
            mine = [e for e in entries if offered[e][0] is h]
            if mine:
                plan.append((h, tuple(offered[e][1] for e in mine), tuple(mine)))
        return tuple(plan)

    def per_step(self, unit, n, plan):
        """{entry: value} of one step for a plan of per_step_plan (each hook asked once, for its names)."""
        out = {}
        for h, names, entries in plan:
            vals = h.per_step(unit, n, names)
            for name, e in zip(names, entries):
                if name not in vals:
                    raise HookError(f"hook {h.name!r} gave no value for its per-step field {name!r}")
                out[e] = vals[name]
        return out

    # ---------------------------------------------------------------------------------------------- run time
    def reset(self, unit):
        for h in self.hooks:
            h.reset(unit)

    def set_world_state(self, unit, state):
        state = dict(state or {})
        unknown = set(state) - {h.name for h in self.hooks}
        if unknown:
            raise HookError(f"world state for hooks that do not exist: {sorted(unknown)}")
        for h in self.hooks:
            if h.name in state:
                h.set_world_state(unit, state[h.name])

    def pre_step(self, unit, n):
        for h in self.hooks:
            h.pre_step(unit, n)

    def post_step(self, unit, n):
        for h in self.hooks:
            h.post_step(unit, n)

    def events(self):
        out = []
        for h in self.hooks:
            for e in h.events():
                e = dict(e)
                e.setdefault("hook", h.name)
                out.append(e)
        return out

    def truth(self, unit):
        return {h.name: h.truth(unit) for h in self.hooks}

    def state(self):
        return {h.name: h.state() for h in self.hooks}

    def load_state(self, s):
        if set(s) != {h.name for h in self.hooks}:
            raise HookError("the hook state is not of this hook set")
        for h in self.hooks:
            h.load_state(s[h.name])

    def model_fields(self, unit):
        return {h.name: h.model_fields(unit) for h in self.hooks}

    def load_model_fields(self, unit, fields):
        if set(fields) != {h.name for h in self.hooks}:
            raise HookError("the hook model fields are not of this hook set")
        for h in self.hooks:
            h.load_model_fields(unit, fields[h.name])
