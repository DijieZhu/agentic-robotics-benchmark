"""The drive's torque envelope E(w) and current limit T_lim per module. PUBLIC.

Pure functions and one small vectorised class; no state of a unit, no MuJoCo, no random numbers. Imports only numpy and
the public reader of `params_public.json` (`state.public_params`); nothing of the law, the registry, the truth channel,
the world or the recorder.

What is computed:

- the drive's torque constant KT_i = 1.5 x NPP x Flux_i x Gr x GREF with the module's OWN registers, evaluated left to
  right as ((((1.5 x NPP) x Flux) x Gr) x GREF), and the current limit T_lim,i = (OC_Value_i x Imax_i) x KT_i (the
  order of the printed values of params_public.json, so that catalogue registers give its numbers bit for bit);
- E(w), w >= 0 in rad/s: the torque the drive can deliver while turning at w in the direction of the torque, in drive
  torque (= output torque on the vendor's fixture + the fixture friction c0 + c1 w). Every edition's E is concave and
  piecewise linear, so it is evaluated as
        E(w) = max(0, min(T_lim, A1 + B1 w, A2 + B2 w))
  with two lines per module (+, x, min, max only; no interpolation routine, no transcendental):
    * DM4340 first edition: the lines through its three published points (52.5, 2.404), (36, 15.334), (27.5, 19.540)
      rpm / N m; above 52.5 rpm along the first segment and clipped at 0, below 27.5 rpm along the last (33.15 N m at
      standstill), never above T_lim;
    * DM8009P and DM4310 V1.1: E = T_lim from standstill to the test speed (100 / 120 rpm), then the straight line
      from (test speed, T_lim) to (no-load speed, fixture friction there), continued and clipped at 0 above (the
      maintainers' own choice);
    * DM4340 V1.1 and DM4310 V1.2: one straight line through two points of the vendor's published speed-torque plot,
      held at T_lim. The points must have been OPENED: this module reads them from the entry
      `opened_points_rpm_Nm` of the edition's envelope in params_public.json (the first and the last plotted point,
      in drive torque). Until that entry exists the edition is refused (`EnvelopeNotOpened`).
- the limits of the in-drive law: L_plus = min(T_lim, E(max(s qdot, 0))), L_minus = min(T_lim, E(max(-s qdot,
  0))). This formula is the law, in every quadrant. In the braking quadrant (torque against motion) it gives
  L = min(T_lim, E(0)), the same limit as at standstill, so the limit is continuous through zero speed. E(0) equals
  T_lim on the DM8009P and the DM4310 V1.1 and is below it on the other three editions (with catalogue registers:
  33.148 against 33.443 N m on the first-edition DM4340, 48.269 against 59.301 on the DM4340 V1.1, 12.613 against
  15.170 on the DM4310 V1.2). The braking limit is never T_lim where E(0) is lower. (With the default mappings no
  joint and no reply meets either figure on these editions: the delivered clamp is 27 N m on the DM4340 editions and
  7 N m on the DM4310, and the reply's torque field ends at 28 and 10 N m.)
  `braking_at_t_lim=True` is an option for comparison only (it puts T_lim in the braking quadrant); the unit never
  sets it.
"""
import numpy as np

from .state import public_params

PI = 3.141592653589793
RPM_TO_RAD_S = PI / 30.0          # 1 rpm = 2 pi / 60 rad/s
EDITIONS = ("DM8009P", "DM4340_1", "DM4340_V1_1", "DM4310_V1_1", "DM4310_V1_2")


class EnvelopeError(ValueError):
    """An edition, register or speed that the envelope does not allow."""


class EnvelopeNotOpened(EnvelopeError):
    """An edition whose speed-torque points have not been opened (status LISTED in params_public.json: named, not
    yet read)."""


# --------------------------------------------------------------------------------------------- torque constant and limit
def kt(npp, flux, gr, gref=1.0):
    """KT = 1.5 x NPP x Flux x Gr x GREF (N m/A), evaluated left to right. Accepts scalars or arrays."""
    return 1.5 * npp * flux * gr * gref


def t_lim(oc_value, imax, kt_value):
    """T_lim = OC_Value x Imax x KT (N m), evaluated as (OC_Value x Imax) x KT."""
    return oc_value * imax * kt_value


def kt_from_registers(regs):
    """KT of one module from its register table (a mapping RID -> value: 16 NPP, 19 Flux, 20 Gr, 30 GREF)."""
    gref = regs.get(30, 1.0)
    if gref is None:
        gref = 1.0
    return kt(float(regs[16]), float(regs[19]), float(regs[20]), float(gref))


def t_lim_from_registers(regs, imax):
    """T_lim of one module from its registers (3 OC_Value and the KT registers) and its current base Imax."""
    return t_lim(float(regs[3]), float(imax), kt_from_registers(regs))


# ----------------------------------------------------------------------------------------------------------- the lines
def fixture_friction(edition, w, public=None):
    """The fixture friction c0 + c1 w (N m) of an edition at w rad/s (the maintainers' own choice, the arm joint
    class's nominal)."""
    p = public_params() if public is None else public
    ff = p["editions"][edition]["fixture_friction"]["value"]
    return ff["c0_Nm"] + ff["c1_Nm_s_per_rad"] * w


def _line(w1, e1, w2, e2):
    """(A, B) of the line through (w1, e1) and (w2, e2), speeds in rad/s."""
    b = (e2 - e1) / (w2 - w1)
    return e1 - b * w1, b


def edition_lines(edition, t_lim_value, public=None):
    """The two lines (A1, B1, A2, B2) of E for one module of `edition` whose current limit is `t_lim_value` (N m);
    speeds in rad/s. E(w) = max(0, min(T_lim, A1 + B1 w, A2 + B2 w))."""
    p = public_params() if public is None else public
    if edition not in EDITIONS:
        raise EnvelopeError(f"unknown edition {edition!r}")
    env = p["envelope"][edition]
    tl = float(t_lim_value)
    if not (tl > 0.0):
        raise EnvelopeError("T_lim must be positive")
    if edition == "DM4340_1":
        (r1, e1), (r2, e2), (r3, e3) = env["points_rpm_Nm"]["value"]
        a1, b1 = _line(r1 * RPM_TO_RAD_S, e1, r2 * RPM_TO_RAD_S, e2)
        a2, b2 = _line(r2 * RPM_TO_RAD_S, e2, r3 * RPM_TO_RAD_S, e3)
        return a1, b1, a2, b2
    if edition in ("DM8009P", "DM4310_V1_1"):
        test = env["test_speed_rpm"]["value"] * RPM_TO_RAD_S
        r_nl, e_nl = env["no_load_point_rpm_Nm"]["value"]
        a, b = _line(test, tl, r_nl * RPM_TO_RAD_S, e_nl)
        return tl, 0.0, a, b
    # DM4340_V1_1, DM4310_V1_2: a straight line through the opened points, held at T_lim
    if "opened_points_rpm_Nm" not in env:
        raise EnvelopeNotOpened(f"the speed-torque points of {edition} are LISTED, not opened")
    (r1, e1), (r2, e2) = env["opened_points_rpm_Nm"]["value"]
    a, b = _line(r1 * RPM_TO_RAD_S, e1, r2 * RPM_TO_RAD_S, e2)
    return a, b, a, b


def envelope_scalar(edition, t_lim_value, w, public=None):
    """E(w) of one module (Python float; w >= 0 in rad/s)."""
    if w < 0.0:
        raise EnvelopeError("E is defined for w >= 0 (the speed in the direction of the torque)")
    a1, b1, a2, b2 = edition_lines(edition, t_lim_value, public)
    return max(0.0, min(min(float(t_lim_value), a1 + b1 * w), a2 + b2 * w))


# ---------------------------------------------------------------------------------------------- the vectorised envelope
class ModuleEnvelopes:
    """E and the law's limits for N modules at once (element-wise ufuncs on whole arrays).

    editions: N edition keys (module order); t_lims: N current limits in N m (from each module's own
    registers and current base: `t_lim_from_registers`)."""

    def __init__(self, editions, t_lims, public=None, braking_at_t_lim=False):
        eds = tuple(editions)
        tl = np.array([float(v) for v in t_lims], dtype=np.float64)
        if len(eds) != tl.shape[0]:
            raise EnvelopeError("one T_lim per module")
        lines = np.array([edition_lines(e, t, public) for e, t in zip(eds, tl)], dtype=np.float64).reshape(len(eds), 4)
        self.editions = eds
        self.t_lim = tl
        self.a1 = lines[:, 0].copy()
        self.b1 = lines[:, 1].copy()
        self.a2 = lines[:, 2].copy()
        self.b2 = lines[:, 3].copy()
        self.braking_at_t_lim = bool(braking_at_t_lim)
        self._e0 = self.E(np.zeros(len(eds)))       # E(0) <= T_lim by construction (min with T_lim inside E)

    def E(self, w):
        """E(w) per module, w >= 0 rad/s (an array of N)."""
        w = np.asarray(w, dtype=np.float64)
        return np.maximum(0.0, np.minimum(np.minimum(self.t_lim, self.a1 + self.b1 * w), self.a2 + self.b2 * w))

    def limits(self, s_qdot):
        """(L_plus, L_minus) of the in-drive law for the speeds s qdot (rad/s, drive sign) of the N modules."""
        # E is evaluated once, at |s qdot|; the other side is E(0) (formed once at construction). For v > 0,
        # E(max(v, 0)) = E(|v|) and E(max(-v, 0)) = E(0), and the reverse for v < 0: the same bits as the formula.
        v = np.asarray(s_qdot, dtype=np.float64)
        e_moving = self.E(np.abs(v))
        pos = v > 0.0
        neg = v < 0.0
        l_plus = np.where(pos, e_moving, self._e0)
        l_minus = np.where(neg, e_moving, self._e0)
        if self.braking_at_t_lim:
            l_plus = np.where(neg, self.t_lim, l_plus)
            l_minus = np.where(pos, self.t_lim, l_minus)
        return l_plus, l_minus


def module_envelopes(edition_per_module, registers, imax, public=None, braking_at_t_lim=False):
    """ModuleEnvelopes from the modules' editions, register tables (a RegisterSet or a sequence of RID mappings) and
    current bases."""
    mods = registers.modules if hasattr(registers, "modules") else tuple(registers)
    tls = [t_lim_from_registers(m, i) for m, i in zip(mods, imax)]
    return ModuleEnvelopes(edition_per_module, tls, public, braking_at_t_lim)


def free_speed(edition, t_lim_value, public=None, friction=None):
    """The speed (rad/s) at which E(w) equals the fixture friction c0 + c1 w (the free speed of a joint), by
    bisection on the piecewise-linear difference (offline use: checks and documents). `friction` = (c0, c1)
    overrides the edition's fixture friction (for instance with J4's own friction)."""
    p = public_params() if public is None else public
    if friction is None:
        ff = p["editions"][edition]["fixture_friction"]["value"]
        c0, c1 = ff["c0_Nm"], ff["c1_Nm_s_per_rad"]
    else:
        c0, c1 = friction
    lo, hi = 0.0, 1000.0
    f = lambda w: envelope_scalar(edition, t_lim_value, w, p) - (c0 + c1 * w)
    if f(lo) <= 0.0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
