"""Bundled material database with provenance.

Every entry carries the document it was read from. Constants published
in US customary units (in/cycle, ksi sqrt(in)) are converted to SI
(m/cycle, MPa sqrt(m)) on load:

  1 ksi sqrt(in) = 1.0988 MPa sqrt(m)
  C_si = C_us * 0.0254 / 1.0988^n   (da/dN = C dK^n in each system)

The database is reference data, not design data. The source documents
are public; check them, and substitute your own basis values for
anything that matters.
"""

from __future__ import annotations

import json
from importlib import resources

from .fracture import ParisLaw, WalkerLaw
from .nasgro import NasgroLaw

KSI_SQRT_IN = 1.0988  # MPa sqrt(m)
KSI = 6.8948          # MPa
INCH = 0.0254         # m


def _raw():
    path = resources.files("damocles").joinpath("data/materials.json")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("_comment", None)
    return data


def available():
    return sorted(_raw())


def get(name):
    """Return the material record in SI units, with the source intact."""
    data = _raw()
    if name not in data:
        raise KeyError(f"unknown material {name!r}, available: {available()}")
    rec = json.loads(json.dumps(data[name]))  # deep copy
    if rec.pop("units") == "us":
        _convert_us_to_si(rec)
    return rec


def _convert_us_to_si(rec):
    if "nasgro" in rec:
        g = rec["nasgro"]
        g["c"] = g["c"] * INCH / KSI_SQRT_IN ** g["n"]
        g["dk1"] = g["dk1"] * KSI_SQRT_IN
    if "walker" in rec:
        w = rec["walker"]
        w["c"] = w["c"] * INCH / KSI_SQRT_IN ** w["n"]
    if "paris" in rec:
        p = rec["paris"]
        p["c"] = p["c"] * INCH / KSI_SQRT_IN ** p["m"]
        p["dk_threshold"] = p.get("dk_threshold", 0.0) * KSI_SQRT_IN
    if "toughness" in rec:
        t = rec["toughness"]
        for key in ("kic", "kc"):
            if key in t and t[key] is not None:
                t[key] = t[key] * KSI_SQRT_IN
    if "strength" in rec:
        s = rec["strength"]
        for key in ("ys", "uts"):
            if key in s and s[key] is not None:
                s[key] = s[key] * KSI


def growth_law(name, kind=None, c_override=None):
    """Build a growth law (SI units) from a database entry.

    kind defaults to the best model the entry carries: nasgro, then
    walker, then paris. c_override substitutes a per-sample array for
    the coefficient, which is how growth scatter enters a study.
    """
    rec = get(name)
    if kind is None:
        kind = next(k for k in ("nasgro", "walker", "paris") if k in rec)
    if kind not in rec:
        raise KeyError(f"{name} has no {kind!r} fit, has: "
                       f"{[k for k in ('nasgro', 'walker', 'paris') if k in rec]}")
    g = rec[kind]
    c = g["c"] if c_override is None else c_override
    if kind == "nasgro":
        return NasgroLaw(c=c, n=g["n"], p=g["p"], q=g["q"], dk1=g["dk1"],
                         cth_plus=g["cth_plus"], cth_minus=g["cth_minus"],
                         alpha=g["alpha"], smax_sigma0=g["smax_sigma0"])
    if kind == "walker":
        # NASGRO's Walker m relates to our gamma directly
        return WalkerLaw(c=c, m=g["n"], gamma=g["m_plus"])
    return ParisLaw(c=c, m=g["m"], dk_threshold=g.get("dk_threshold", 0.0))
