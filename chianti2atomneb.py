#!/usr/bin/env python3
"""
Convert a local CHIANTI ASCII database tree into AtomNeb-style FITS files:

    AtomAij.fits   transition probabilities Aij
    AtomElj.fits   energy levels Ej
    AtomOmij.fits  effective collision strengths Omega/Upsilon(T)

The default FITS layout matches AtomNeb-py's atomic-data/chianti90:

    PRIMARY
    References
    <ion>_aij_<TAG>     in AtomAij.fits
    <ion>_elj           in AtomElj.fits
    <ion>_omij_<TAG>    in AtomOmij.fits

Use --include-list if you also want an extra List HDU after References.

Input-format support:
    CHIANTI 6/7 and Cloudy CHIANTI 9:  .elvlc old level format, .wgfa old radiative format,
                        .splups old one-line scaled upsilon format
    CHIANTI 8/9 original/10/11:      .elvlc compact level format, .wgfa with trailing
                        transition labels, .scups three-line scaled upsilon format

Examples of extension names:

    h_i_aij_CHI_CUSTOM
    h_i_elj
    h_i_omij_CHI_CUSTOM
    o_iii_aij_CHI_CUSTOM
    o_iii_elj
    o_iii_omij_CHI_CUSTOM

The ion extensions are sorted in atomic-number order, then ion stage, matching
AtomNeb's convention more closely than alphabetic sorting.

Requirements:
    pip install numpy astropy

Example:
    python chianti_to_atomneb_fits.py \
      --chianti-root ./chianti \
      --out-dir ./atomic-data/chianti_custom \
      --tag CHI_CUSTOM \
      --temp-log10-min 2 --temp-log10-max 9 --temp-points 71

For one ion only:
    python chianti_to_atomneb_fits.py \
      --chianti-root ./chianti \
      --ions o_3 \
      --out-dir ./test_o3 \
      --tag CHI_CUSTOM
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from astropy.io import fits
except Exception as exc:  # pragma: no cover
    raise SystemExit("ERROR: this converter needs astropy. Install it with: pip install astropy") from exc

RYD_K = 157887.500  # Rydberg energy / k_B, K

ELEMENT_Z: Dict[str, int] = {
    "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8,
    "f": 9, "ne": 10, "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15,
    "s": 16, "cl": 17, "ar": 18, "k": 19, "ca": 20, "sc": 21, "ti": 22,
    "v": 23, "cr": 24, "mn": 25, "fe": 26, "co": 27, "ni": 28, "cu": 29,
    "zn": 30,
}

ROMAN: Dict[int, str] = {
    1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi", 7: "vii", 8: "viii",
    9: "ix", 10: "x", 11: "xi", 12: "xii", 13: "xiii", 14: "xiv", 15: "xv",
    16: "xvi", 17: "xvii", 18: "xviii", 19: "xix", 20: "xx", 21: "xxi",
    22: "xxii", 23: "xxiii", 24: "xxiv", 25: "xxv", 26: "xxvi", 27: "xxvii",
    28: "xxviii", 29: "xxix", 30: "xxx", 31: "xxxi",
}


@dataclass
class Level:
    index: int
    configuration: str
    term: str
    J_text: str
    J_value: float
    g: float
    energy_cm: float


@dataclass
class SplupsRecord:
    lower: int
    upper: int
    ttype: int
    gf: float
    de_ryd: float
    cups: float
    spline_values: np.ndarray
    scaled_t: Optional[np.ndarray] = None
    source_format: str = "splups"

    @property
    def npts(self) -> int:
        return int(len(self.spline_values))


@dataclass
class IonData:
    ion_id: str
    ion_dir: Path
    levels: List[Level]
    aij: np.ndarray
    splups: List[SplupsRecord]
    ref_elj: str
    ref_aij: str
    ref_omij: str


def normalize_ion_id(s: str) -> str:
    s = s.strip().lower().replace("-", "_")
    if not re.match(r"^[a-z]+_[0-9]+$", s):
        raise ValueError(f"Invalid CHIANTI ion id '{s}'; expected e.g. o_3, fe_17")
    return s


def ion_sort_key(ion_id: str) -> Tuple[int, int, str]:
    el, stage = normalize_ion_id(ion_id).split("_")
    return (ELEMENT_Z.get(el, 999), int(stage), el)


def ion_ext_prefix(ion_id: str) -> str:
    el, stage = normalize_ion_id(ion_id).split("_")
    return f"{el}_{ROMAN.get(int(stage), stage)}"


def aij_extname(ion_id: str, tag: str) -> str:
    return f"{ion_ext_prefix(ion_id)}_aij_{tag}"


def elj_extname(ion_id: str) -> str:
    return f"{ion_ext_prefix(ion_id)}_elj"


def omij_extname(ion_id: str, tag: str) -> str:
    return f"{ion_ext_prefix(ion_id)}_omij_{tag}"


def is_comment_or_end(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("%") or s.startswith("-1")


def clean_chianti_reference_text(text: str) -> str:
    """Clean CHIANTI bibliography/comment text into a pure reference string.

    CHIANTI reference blocks often contain descriptive prefixes before the
    actual citation, for example ``% A values:``, ``Transition data from:``,
    ``Experimental energies from:``, or ``Effective Collision Strengths and
    gf-values from``.  AtomNeb's References tables store the citation text,
    so this function removes the descriptive prefix while preserving the
    bibliographic reference itself.
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    # Strip CHIANTI comment markers and normalize whitespace/FITS-safe ASCII.
    s = s.lstrip("%#! ").strip()
    s = fits_ascii(s) if "fits_ascii" in globals() else s
    s = " ".join(s.split())

    # Remove common "X: reference" prefixes.  Keep the right-hand side.
    colon_prefix_re = re.compile(
        r"^(?:"
        r"a[- ]?values?[^:]{0,160}|transition probabilities?[^:]{0,160}|"
        r"transition data[^:]{0,160}|radiative (?:rates?|data)[^:]{0,160}|"
        r"oscillator strengths?[^:]{0,160}|gf(?:[- ]?values?)?[^:]{0,160}|"
        r"effective collision strengths?[^:]{0,160}|collision strengths?[^:]{0,160}|"
        r"collisional data[^:]{0,160}|all collisional data[^:]{0,160}|"
        r"upsilons?[^:]{0,160}|excitation data(?: from)?[^:]{0,160}|"
        r"data(?: from)?[^:]{0,160}|outer shell[^:]{0,160}|inner shell[^:]{0,160}|"
        r"wavelengths?[^:]{0,160}|"
        r"experimental energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"theoretical energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"observed energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"energy levels?[^:]{0,160}|levels? [^:]{0,160}|"
        r"references?[^:]{0,160}|source[^:]{0,160}|note[^:]{0,160}"
        r")\s*:\s*",
        re.IGNORECASE,
    )

    # Remove common "X from reference" prefixes without a colon.
    from_prefix_re = re.compile(
        r"^(?:"
        r"a[- ]?values?[^:]{0,160}|transition probabilities?[^:]{0,160}|"
        r"transition data[^:]{0,160}|radiative (?:rates?|data)[^:]{0,160}|"
        r"oscillator strengths?[^:]{0,160}|gf(?:[- ]?values?)?[^:]{0,160}|"
        r"effective collision strengths?[^:]{0,160}|collision strengths?[^:]{0,160}|"
        r"collisional data[^:]{0,160}|all collisional data[^:]{0,160}|"
        r"excitation data[^:]{0,160}|data[^:]{0,160}|"
        r"experimental energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"theoretical energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"observed energ(?:y|ies)(?: levels?)?[^:]{0,160}|"
        r"energy levels?[^:]{0,160}|wavelengths?[^:]{0,160}"
        r")\s+(?:are\s+)?from\s+",
        re.IGNORECASE,
    )

    # Remove a few left-side qualifiers that often precede "are/is from".
    are_from_re = re.compile(
        r"^(?:"
        r"theoretical energies,?\s*a(?:-|\s+and\s+)gf values?\s+for\s+levels?\s+[^,;:]{0,120}|"
        r"a[- ]?values?\s+for\s+levels?\s+[^,;:]{0,120}|"
        r"gf[- ]?values?\s+for\s+levels?\s+[^,;:]{0,120}|"
        r"levels?\s+[^,;:]{0,120}|for\s+levels?\s+[^,;:]{0,120}|"
        r"up\s+to\s+level\s+[^,;:]{0,120}|"
        r"ground\s+(?:conf\.?|configuration)[^,;:]{0,120}"
        r")\s+(?:are|is)\s+from\s+",
        re.IGNORECASE,
    )

    for _ in range(8):
        old = s
        s = colon_prefix_re.sub("", s).strip()
        s = from_prefix_re.sub("", s).strip()
        s = are_from_re.sub("", s).strip()
        s = re.sub(r"^for\s+levels?\s+[^,;:]{0,120}?\s+(?:are\s+)?from\s+", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^levels?\s+[^,;:]{0,120}?\s*:\s*data\s+from\s+", "", s, flags=re.IGNORECASE).strip()
        if s == old:
            break

    # Drop leading low-content qualifiers left behind by some CHIANTI comments.
    s = re.sub(
        r"^(?:ground conf\.?|ground configuration|up to level [^:,.]+|levels? [^:,.]+)\s*[:;,.-]*\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()

    # Remove a leading comma/colon/semicolon introduced by prefix stripping.
    s = s.strip(" ;,:")
    return s



def extract_chianti_reference(path: Path, kind: str = "generic") -> str:
    """Extract and clean the most relevant bibliographic reference.

    ``kind`` selects a preferred reference when a CHIANTI file has several
    comments.  For example, .wgfa files may contain both energy-level and
    A-value references; for AtomAij.fits we prefer A-value/transition-data
    references rather than the first energy-level citation.
    """
    if not path.exists():
        return ""

    kind = (kind or "generic").lower()
    if kind in {"aij", "wgfa"}:
        preferred = re.compile(
            r"(a[- ]?values?|transition\s+data|transition\s+probabilit|"
            r"radiative\s+(?:rates?|data)|oscillator\s+strengths?|gf[- ]?values?)",
            re.IGNORECASE,
        )
        avoid = re.compile(r"(energy\s+levels?|observed\s+energ|theoretical\s+energ|experimental\s+energ)", re.IGNORECASE)
    elif kind in {"elj", "elvlc"}:
        preferred = re.compile(r"(energy\s+levels?|observed\s+energ|theoretical\s+energ|experimental\s+energ|wavelengths?)", re.IGNORECASE)
        avoid = re.compile(r"(collision|excitation|upsilon|effective collision|a[- ]?values?|oscillator|gf[- ]?values?)", re.IGNORECASE)
    elif kind in {"omij", "splups", "scups"}:
        preferred = re.compile(r"(effective\s+collision|collision\s+strength|collisional\s+data|excitation\s+data|upsilon)", re.IGNORECASE)
        avoid = re.compile(r"(energy\s+levels?|a[- ]?values?|oscillator|gf[- ]?values?|transition\s+data)", re.IGNORECASE)
    else:
        preferred = re.compile(r".", re.IGNORECASE)
        avoid = re.compile(r"$^")

    lines = path.read_text(errors="ignore").splitlines()
    try:
        first_end = next(i for i, line in enumerate(lines) if line.strip().startswith("-1"))
    except StopIteration:
        first_end = -1

    skip_prefixes = (
        "doi:", "note:", "produced as", "chianti", "http://", "https://",
        "only ", "the rates", "are from", "mchf", "the mao",
        "at the highest", "have been retained", "nist atomic",
    )

    candidates: List[Tuple[int, str]] = []
    for raw in lines[first_end + 1:]:
        s0 = raw.strip()
        if not s0 or s0.startswith("-1"):
            continue
        low0 = s0.lstrip("%#! ").strip().lower()
        if low0.endswith(":"):
            continue
        if any(low0.startswith(p) for p in skip_prefixes):
            continue

        cleaned = clean_chianti_reference_text(s0)
        if not cleaned:
            continue
        low = cleaned.lower()
        if any(low.startswith(p) for p in skip_prefixes):
            continue

        looks_like_citation = bool(re.search(r"\b(18|19|20)\d{2}\b", cleaned)) or ("," in cleaned)
        if not looks_like_citation:
            continue

        raw_is_preferred = bool(preferred.search(s0))
        raw_is_avoided = bool(avoid.search(s0))
        # Lower score is better.
        score = 0
        if not raw_is_preferred:
            score += 10
        if raw_is_avoided and not raw_is_preferred:
            score += 20
        if not re.search(r"\b(18|19|20)\d{2}\b", cleaned):
            score += 5
        if "," not in cleaned:
            score += 2
        candidates.append((score, cleaned))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return f"CHIANTI ASCII file: {path.name}"



def _is_float_token(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _format_j_text(J_value: float) -> str:
    if not np.isfinite(J_value):
        return ""
    if abs(J_value - int(J_value)) < 1e-6:
        return str(int(J_value))
    return f"{J_value:g}"


def parse_elvlc(path: Path, max_levels: Optional[int] = None) -> List[Level]:
    """Read CHIANTI .elvlc level data from CHIANTI 9/10 and CHIANTI 11.

    CHIANTI .elvlc files are not fully column-compatible across ions/releases.
    This parser handles the fixed CHIANTI 9/10 form and several CHIANTI 11
    compact forms by searching for the physical pattern

        multiplicity  L-symbol  J  Eobs_cm  Eth_cm

    and then ignoring any additional trailing columns such as:
        - Rydberg energy columns
        - statistical weight / n / l quantum labels
        - jj-coupling or LS-coupling text labels, e.g. "3s [3/2]2"

    Known supported examples:
        CHIANTI 9/10:
            idx config... mult Lint Lsym J g Eobs_cm Eobs_Ryd Eth_cm Eth_Ryd

        CHIANTI 11 compact H-like:
            idx config mult Lsym J Eobs_cm Eth_cm

        CHIANTI 11 with trailing quantum columns, e.g. N IV:
            idx config... mult Lsym J Eobs_cm Eth_cm g n l

        CHIANTI 11 with trailing labels, e.g. Ne I:
            idx config mult Lsym J Eobs_cm Eth_cm label...

        CHIANTI 11 with configuration index and Rydberg columns, e.g. O VI:
            idx config... cfg_index mult Lsym J Eobs_cm Eth_cm Eobs_Ryd Eth_Ryd

    AtomNeb only needs configuration, term, J, statistical weight, and an
    energy in cm^-1.
    """
    def clean_tokens(line: str) -> List[str]:
        return line.replace(",", " ").split()

    def is_intlike(x: str) -> bool:
        return bool(re.fullmatch(r"[+-]?\d+(?:\.0+)?", str(x)))

    def is_floatlike(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    def clean_config(parts: Sequence[str]) -> str:
        parts = list(parts)
        # Older CHIANTI 6/7-style rows can include a configuration-index column
        # immediately after the level index, e.g.
        #   idx cfg_index configuration mult Lint L J g ...
        # Some CHIANTI 11 ions include a configuration-index column immediately
        # before multiplicity, e.g. "1s2.2s  1  2 S 0.5 ...".  Neither index
        # is part of the spectroscopic configuration used by AtomNeb.
        if len(parts) > 1 and re.fullmatch(r"[+-]?\d+", parts[0] or ""):
            parts = parts[1:]
        if len(parts) > 1 and re.fullmatch(r"[+-]?\d+", parts[-1] or ""):
            parts = parts[:-1]
        return " ".join(parts).strip() or "-"

    def term_ok(mult: str, Lsym: str) -> bool:
        if not is_intlike(mult):
            return False
        # Allow standard spectroscopic letters through high-L labels.
        return bool(re.fullmatch(r"[A-Za-z]+", str(Lsym)))

    def add_level(idx: int, configuration: str, mult: str, Lsym: str,
                  J_value: float, g: float, e_obs_cm: float, e_th_cm: float) -> None:
        energy_cm = e_obs_cm if np.isfinite(e_obs_cm) and e_obs_cm > 0 else e_th_cm
        if not np.isfinite(energy_cm):
            energy_cm = 0.0
        if not np.isfinite(g) or g <= 0:
            g = max(2.0 * J_value + 1.0, 1.0) if np.isfinite(J_value) else 1.0
        try:
            mult_txt = str(int(float(mult))) if float(mult).is_integer() else str(mult)
        except Exception:
            mult_txt = str(mult)
        term = f"{mult_txt}{Lsym}"
        levels.append(Level(
            index=idx,
            configuration=configuration,
            term=term,
            J_text=_format_j_text(J_value),
            J_value=float(J_value),
            g=float(g),
            energy_cm=float(energy_cm),
        ))

    levels: List[Level] = []
    for raw in path.read_text(errors="ignore").splitlines():
        if is_comment_or_end(raw):
            continue
        p = clean_tokens(raw)
        if len(p) < 6:
            continue
        try:
            idx = int(p[0])
        except Exception:
            continue
        if max_levels is not None and idx > max_levels:
            continue

        parsed = False

        # CHIANTI 9/10 old style: idx config... mult Lint Lsym J g Eobs_cm
        # Eobs_Ryd Eth_cm Eth_Ryd.  Keep this first because the extra Lint
        # column means the generic CHIANTI 11 pattern should not be used.
        if len(p) >= 12:
            try:
                J9 = float(p[-6])
                g9 = float(p[-5])
                eobs9 = float(p[-4])
                eth9 = float(p[-2])
                mult = p[-9]
                Lsym = p[-7]
                if term_ok(mult, Lsym) and g9 > 0.0 and abs(g9 - (2.0 * J9 + 1.0)) < 1.0e-4:
                    add_level(idx, clean_config(p[1:-9]), mult, Lsym, J9, g9, eobs9, eth9)
                    parsed = True
            except Exception:
                parsed = False

        # Generic CHIANTI 11 compact parser.  Search left-to-right for:
        #   multiplicity, L-symbol, J, Eobs_cm, Eth_cm
        # This handles compact H-like rows, trailing labels, trailing g/n/l
        # quantum columns, and the O VI style with a configuration-index column.
        if not parsed:
            for i in range(1, max(1, len(p) - 4)):
                if not term_ok(p[i], p[i + 1]):
                    continue
                if not (is_floatlike(p[i + 2]) and is_floatlike(p[i + 3]) and is_floatlike(p[i + 4])):
                    continue
                try:
                    J11 = float(p[i + 2])
                    eobs11 = float(p[i + 3])
                    eth11 = float(p[i + 4])
                except Exception:
                    continue

                # Reject implausible matches from the middle of text labels.
                if not np.isfinite(J11) or J11 < 0.0:
                    continue
                if not np.isfinite(eobs11) or not np.isfinite(eth11):
                    continue

                g11 = 2.0 * J11 + 1.0
                # Some CHIANTI 11 variants put g immediately after Eth_cm.
                # Use it if it is numerically consistent; otherwise ignore it
                # because the token may be n, l, a label, or Rydberg energy.
                if i + 5 < len(p) and is_floatlike(p[i + 5]):
                    try:
                        maybe_g = float(p[i + 5])
                        if maybe_g > 0.0 and abs(maybe_g - g11) < 1.0e-4:
                            g11 = maybe_g
                    except Exception:
                        pass

                add_level(idx, clean_config(p[1:i]), p[i], p[i + 1], J11, g11, eobs11, eth11)
                parsed = True
                break

        if not parsed:
            continue

    if not levels:
        raise RuntimeError(f"No levels read from {path}")
    levels.sort(key=lambda x: x.index)
    return levels


def parse_wgfa(path: Path, nlevels: int) -> np.ndarray:
    """Read CHIANTI .wgfa radiative transition probabilities.

    CHIANTI 9/10 rows contain five numeric columns:
        lower upper wavelength_A gf A_value

    CHIANTI 11 rows keep those same first five fields but append a textual
    transition label, e.g.:
        lower upper wavelength_A gf A_value   lower-term - upper-term

    Therefore this parser only consumes the first five fields and explicitly
    ignores any trailing label text.  The resulting AtomNeb matrix is stored as
    row=lower level-1, column=upper level-1.
    """
    aij = np.zeros((nlevels, nlevels), dtype=np.float64)
    for raw in path.read_text(errors="ignore").splitlines():
        if is_comment_or_end(raw):
            continue
        p = raw.split()
        if len(p) < 5:
            continue
        try:
            lower = int(p[0])
            upper = int(p[1])
            # p[2] wavelength and p[3] gf are not needed for AtomAij, but
            # parsing them validates that the row is a real data row.  CHIANTI
            # 11 may append term labels after p[4], so do not parse past p[4].
            _wavelength_A = float(p[2])
            _gf = float(p[3])
            aval = float(p[4])
        except Exception:
            continue
        if 1 <= lower <= nlevels and 1 <= upper <= nlevels:
            aij[lower - 1, upper - 1] = aval
    return aij


def parse_splups(path: Path, max_level: Optional[int] = None) -> List[SplupsRecord]:
    """Read old CHIANTI .splups electron excitation data.

    Old .splups records are one-line rows with columns:
        Z ion lower upper type gf deltaE_ryd C spline_values...

    The spline abscissae are implicit and uniformly spaced from 0 to 1.
    """
    if not path.exists():
        return []
    rows: List[SplupsRecord] = []
    for raw in path.read_text(errors="ignore").splitlines():
        if is_comment_or_end(raw):
            continue
        p = raw.split()
        if len(p) < 9:
            continue
        try:
            lower = int(p[2])
            upper = int(p[3])
            if max_level is not None and (lower > max_level or upper > max_level):
                continue
            rows.append(SplupsRecord(
                lower=lower,
                upper=upper,
                ttype=int(p[4]),
                gf=float(p[5]),
                de_ryd=float(p[6]),
                cups=float(p[7]),
                spline_values=np.array([float(x) for x in p[8:]], dtype=np.float64),
                scaled_t=None,
                source_format="splups",
            ))
        except Exception:
            continue
    return rows


def _parse_float_line(line: str) -> np.ndarray:
    return np.array([float(x) for x in line.split()], dtype=np.float64)


def _read_n_float_values(lines: List[str], i: int, n: int) -> Tuple[np.ndarray, int]:
    """Read at least n float values from one or more CHIANTI array lines.

    Most CHIANTI 11 .scups files put each array on a single line, but this
    helper also tolerates wrapped arrays.  It stops as soon as n values have
    been accumulated and returns the updated line index.
    """
    vals: List[float] = []
    while i < len(lines) and len(vals) < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("-1"):
            break
        # Reference/comment lines after -1 are not part of data blocks.
        if line.startswith("%") or line.startswith("#") or line.startswith("!"):
            i += 1
            continue
        vals.extend(float(x) for x in line.split())
        i += 1
    if len(vals) < n:
        raise ValueError(f"Expected {n} float values in .scups block, got {len(vals)}")
    return np.array(vals[:n], dtype=np.float64), i


def parse_scups(path: Path, max_level: Optional[int] = None) -> List[SplupsRecord]:
    """Read CHIANTI 11+ .scups electron excitation data.

    This is intentionally separate from parse_splups because CHIANTI 11 .scups
    does *not* have the old CHIANTI 9 .splups columns.

    Old .splups one-line format:
        Z ion lower upper type gf deltaE_ryd C spline_values...

    CHIANTI 11 .scups three-block format:
        lower upper deltaE_ryd gf highT_limit npts type C
        scaled_temperature_grid[0:npts]
        scaled_upsilon_grid[0:npts]

    The highT_limit value is stored in the source file for CHIANTI bookkeeping.
    To construct AtomNeb-style Omij tables, we need lower/upper, transition
    type, gf, deltaE, C, the explicit scaled-temperature grid, and the scaled
    upsilon values.  The same Burgess--Tully descaling formula is then applied
    after interpolation on the explicit .scups scaled-temperature grid.
    """
    if not path.exists():
        return []

    raw_lines = path.read_text(errors="ignore").splitlines()
    rows: List[SplupsRecord] = []
    i = 0
    while i < len(raw_lines):
        header = raw_lines[i].strip()
        i += 1
        if not header:
            continue
        if header.startswith("-1"):
            break
        if header.startswith("%") or header.startswith("#") or header.startswith("!"):
            continue

        hp = header.split()
        if len(hp) < 8:
            # Not a valid .scups transition header.
            continue
        try:
            # CHIANTI 11 .scups header columns:
            # 0 lower, 1 upper, 2 DeltaE[Ryd], 3 gf,
            # 4 high-T limit, 5 npts, 6 scaling type, 7 C parameter.
            lower = int(hp[0])
            upper = int(hp[1])
            de_ryd = float(hp[2])
            gf = float(hp[3])
            high_t_limit = float(hp[4])  # kept for format clarity; not written to AtomNeb table
            npts = int(hp[5])
            ttype = int(hp[6])
            cups = float(hp[7])
        except Exception:
            continue

        if npts <= 0:
            continue

        try:
            scaled_t, i = _read_n_float_values(raw_lines, i, npts)
            scaled_ups, i = _read_n_float_values(raw_lines, i, npts)
        except Exception:
            continue

        if max_level is not None and (lower > max_level or upper > max_level):
            continue
        rows.append(SplupsRecord(
            lower=lower,
            upper=upper,
            ttype=ttype,
            gf=gf,
            de_ryd=de_ryd,
            cups=cups,
            spline_values=scaled_ups,
            scaled_t=scaled_t,
            source_format="scups",
        ))
    return rows


def read_collision_strengths(
    ion_id: str,
    ion_dir: Path,
    max_level: Optional[int] = None,
    collision_format: str = "auto",
) -> Tuple[List[SplupsRecord], Path, str]:
    """Read electron-impact collision data for one ion.

    Returns (records, source_path, source_format).  In auto mode, .scups is
    preferred when present because it is the CHIANTI 11+ electron-collision
    format; otherwise the reader falls back to old .splups.
    """
    fmt = str(collision_format or "auto").lower()
    splups_path = ion_dir / f"{ion_id}.splups"
    scups_path = ion_dir / f"{ion_id}.scups"

    if fmt == "auto":
        if scups_path.exists():
            recs = parse_scups(scups_path, max_level=max_level)
            return recs, scups_path, "scups"
        if splups_path.exists():
            recs = parse_splups(splups_path, max_level=max_level)
            return recs, splups_path, "splups"
        return [], splups_path, "none"
    if fmt == "scups":
        return parse_scups(scups_path, max_level=max_level), scups_path, "scups"
    if fmt == "splups":
        return parse_splups(splups_path, max_level=max_level), splups_path, "splups"
    raise ValueError(f"Unknown collision format: {collision_format}")


def chianti_upsilon(rec: SplupsRecord, temperature_K: np.ndarray) -> np.ndarray:
    """Descale CHIANTI .splups values to effective collision strengths."""
    T = np.asarray(temperature_K, dtype=np.float64)
    de = max(float(rec.de_ryd), 1e-300)
    c = max(float(rec.cups), 1e-12)
    kte = np.maximum(T / RYD_K / de, 1e-300)
    yarr = np.asarray(rec.spline_values, dtype=np.float64)
    if rec.scaled_t is not None and len(rec.scaled_t) == len(yarr):
        xgrid = np.asarray(rec.scaled_t, dtype=np.float64)
    else:
        xgrid = np.linspace(0.0, 1.0, len(yarr))
    order = np.argsort(xgrid)
    xgrid = np.clip(xgrid[order], 0.0, 1.0)
    yarr = yarr[order]
    ttype = int(rec.ttype)

    if ttype in (1, 4):
        x = 1.0 - np.log(c) / np.log(kte + c)
        x = np.clip(x, 0.0, 1.0)
        y = np.interp(x, xgrid, yarr)
        ups = y * (np.log(kte + math.e) if ttype == 1 else np.log(kte + c))
    elif ttype in (2, 3, 5):
        x = np.clip(kte / (kte + c), 0.0, 1.0)
        y = np.interp(x, xgrid, yarr)
        ups = y if ttype == 2 else y / (kte + 1.0)
    elif ttype == 6:
        x = np.clip(kte / (kte + c), 0.0, 1.0)
        y = np.interp(x, xgrid, yarr)
        ups = 10.0 ** y
    else:
        x = np.clip(kte / (kte + c), 0.0, 1.0)
        ups = np.interp(x, xgrid, yarr)

    return np.where(np.isfinite(ups) & (ups > 0.0), ups, 0.0).astype(np.float64)


def find_ion_dirs(root: Path, selected_ions: Optional[Sequence[str]] = None) -> List[Tuple[str, Path]]:
    selected = {normalize_ion_id(x) for x in selected_ions} if selected_ions else None
    found: Dict[str, Path] = {}
    for elvlc in root.rglob("*.elvlc"):
        ion_id = elvlc.stem.lower()
        if not re.match(r"^[a-z]+_[0-9]+$", ion_id):
            continue
        if selected is not None and ion_id not in selected:
            continue
        ion_dir = elvlc.parent
        if (ion_dir / f"{ion_id}.wgfa").exists():
            found[ion_id] = ion_dir
    missing = sorted((selected or set()) - set(found), key=ion_sort_key)
    if missing:
        raise FileNotFoundError(f"Requested ions not found below {root}: {', '.join(missing)}")
    return sorted(found.items(), key=lambda kv: ion_sort_key(kv[0]))


def read_ion(ion_id: str, ion_dir: Path, max_levels: Optional[int], collision_format: str = "auto") -> IonData:
    elvlc = ion_dir / f"{ion_id}.elvlc"
    wgfa = ion_dir / f"{ion_id}.wgfa"
    levels = parse_elvlc(elvlc, max_levels=max_levels)
    nlevels = max(level.index for level in levels)
    aij = parse_wgfa(wgfa, nlevels=nlevels)
    splups, collision_path, collision_source = read_collision_strengths(
        ion_id, ion_dir, max_level=nlevels, collision_format=collision_format
    )
    return IonData(
        ion_id=ion_id,
        ion_dir=ion_dir,
        levels=levels,
        aij=aij,
        splups=splups,
        ref_elj=extract_chianti_reference(elvlc, kind="elj"),
        ref_aij=extract_chianti_reference(wgfa, kind="aij"),
        ref_omij=extract_chianti_reference(collision_path, kind="omij") if collision_path.exists() else "",
    )


def make_list_hdu(entries: Sequence[Tuple[str, int]], data_column_name: str) -> fits.BinTableHDU:
    """Create AtomNeb-like 'List' binary table.

    entries contains (atomic_data_name, FITS extension number), where extension
    numbers are 1-based HDU indices in the final file.  AtomNeb's files use
    column names AIJ_DATA/EJ_DATA/OMIJ_DATA and EXTENSION.
    """
    if not entries:
        entries = [("", 0)]
    max_name = max(len(data_column_name), max(len(x[0]) for x in entries), 1)
    names = np.array([x[0] for x in entries], dtype=f"S{max_name}")
    exts = np.array([x[1] for x in entries], dtype=np.int16)
    cols = [
        fits.Column(name=data_column_name, format=f"{max_name}A", array=names),
        fits.Column(name="EXTENSION", format="I", array=exts),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    return set_extname_exact(hdu, "List")



def fits_ascii(value: object) -> str:
    """Return an ASCII-safe string for FITS string columns and comments."""
    if value is None:
        return ""
    text = str(value)
    text = (text
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u2212", "-")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"'))
    return unicodedata.normalize("NFKD", text).encode("ascii", "replace").decode("ascii")

def make_references_hdu(entries: Sequence[Tuple[str, str]]) -> fits.BinTableHDU:
    if not entries:
        entries = [("", "")]
    safe_entries = [(fits_ascii(k), fits_ascii(r)) for k, r in entries]
    max_key = max(10, max(len(k) for k, _ in safe_entries))
    max_ref = max(16, max(len(r) for _, r in safe_entries))
    cols = [
        fits.Column(name="ATOMICDATA", format=f"{max_key}A",
                    array=np.array([k for k, _ in safe_entries], dtype=f"S{max_key}")),
        fits.Column(name="REFERENCE", format=f"{max_ref}A",
                    array=np.array([r for _, r in safe_entries], dtype=f"S{max_ref}")),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    return set_extname_exact(hdu, "References")


def add_common_comments(hdu, kind: str, ion: IonData, source_ref: str, tag: str = "", extra: Optional[Dict[str, object]] = None) -> None:
    el, stage = ion.ion_id.split("_")
    hdu.header.add_comment("***********************************")
    hdu.header.add_comment(kind)
    hdu.header.add_comment(f"ATOM: {el.capitalize()}")
    hdu.header.add_comment(f"ION: {ROMAN.get(int(stage), stage).upper()}")
    hdu.header.add_comment(f"N_LEVELS: {ion.aij.shape[0]}")
    if tag:
        hdu.header.add_comment(f"TAG: {tag}")
    if extra:
        for k, v in extra.items():
            hdu.header.add_comment(fits_ascii(f"{k}: {v}"))
    if source_ref:
        hdu.header.add_comment(fits_ascii(f"REFERENCE: {source_ref}"))
    hdu.header.add_comment("***********************************")


def make_aij_hdu(ion: IonData, tag: str) -> fits.ImageHDU:
    name = aij_extname(ion.ion_id, tag)
    hdu = fits.ImageHDU(data=ion.aij.astype(np.float64))
    set_extname_exact(hdu, name)
    add_common_comments(hdu, "Transition Probabilities (Aij)", ion, ion.ref_aij, tag, {"LINES": int(np.count_nonzero(ion.aij))})
    return hdu


def make_elj_hdu(ion: IonData) -> fits.BinTableHDU:
    name = elj_extname(ion.ion_id)
    levels = ion.levels
    ref = np.array([name] * len(levels))
    config = np.array([fits_ascii(x.configuration) for x in levels])
    term = np.array([fits_ascii(x.term) for x in levels])
    jtxt = np.array([fits_ascii(x.J_text) for x in levels])
    jval = np.array([x.J_value for x in levels], dtype=np.float32)
    ej = np.array([x.energy_cm for x in levels], dtype=np.float64)
    w_config = max(1, max(len(x) for x in config))
    w_term = max(1, max(len(x) for x in term))
    w_j = max(1, max(len(x) for x in jtxt))
    w_ref = max(1, len(name))
    cols = [
        fits.Column(name="CONFIGURATION", format=f"{w_config}A", array=config.astype(f"S{w_config}")),
        fits.Column(name="TERM", format=f"{w_term}A", array=term.astype(f"S{w_term}")),
        fits.Column(name="J", format=f"{w_j}A", array=jtxt.astype(f"S{w_j}")),
        fits.Column(name="J_V", format="E", array=jval),
        fits.Column(name="EJ", format="D", array=ej),
        fits.Column(name="REFERENCE", format=f"{w_ref}A", array=ref.astype(f"S{w_ref}")),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    set_extname_exact(hdu, name)
    add_common_comments(hdu, "Energy Levels (Ej)", ion, ion.ref_elj, "", {"LEVELS": len(levels)})
    return hdu


def make_omij_hdu(ion: IonData, tag: str, temperatures: np.ndarray) -> Optional[fits.BinTableHDU]:
    if not ion.splups:
        return None
    name = omij_extname(ion.ion_id, tag)
    nt = len(temperatures)
    level1 = [0]
    level2 = [0]
    strengths = [temperatures.astype(np.float64)]
    for rec in ion.splups:
        level1.append(int(rec.lower))
        level2.append(int(rec.upper))
        strengths.append(chianti_upsilon(rec, temperatures))
    arr_strength = np.vstack(strengths).astype(np.float64)
    cols = [
        fits.Column(name="LEVEL1", format="I", array=np.array(level1, dtype=np.int16)),
        fits.Column(name="LEVEL2", format="I", array=np.array(level2, dtype=np.int16)),
        fits.Column(name="STRENGTH", format=f"{nt}D", array=arr_strength),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    set_extname_exact(hdu, name)
    source_formats = sorted({rec.source_format for rec in ion.splups})
    add_common_comments(
        hdu, "Collision Strengths (Omega_ij)", ion, ion.ref_omij, tag,
        {"TEMP_STEPS": nt, "LINES": len(ion.splups), "SOURCE_FORMAT": "+".join(source_formats)}
    )
    return hdu




def set_extname_exact(hdu, extname: str):
    """Set EXTNAME without astropy's HDU name upper-casing."""
    hdu.header["EXTNAME"] = extname
    return hdu

def primary_hdu() -> fits.PrimaryHDU:
    primary = fits.PrimaryHDU()
    primary.header["ORIGIN"] = "chianti_to_atomneb_fits.py"
    primary.header["COMMENT"] = "AtomNeb-style FITS generated from CHIANTI ASCII data"
    return primary


def write_fits_files(ions: Sequence[IonData], out_dir: Path, tag: str, temperatures: np.ndarray, overwrite: bool = True, include_list: bool = False, include_references: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(tag).strip() or "CHI_CUSTOM"

    # The ion list is expected to already be sorted, but keep this here to make
    # the output deterministic even if write_fits_files is called directly.
    ions = sorted(list(ions), key=lambda ion: ion_sort_key(ion.ion_id))

    # HDU layout:
    #   PRIMARY=0
    #   optional References
    #   optional List
    #   ion data extensions
    #
    # The List.EXTENSION values are FITS HDU numbers using the same 0-based
    # convention shown by fv/CFITSIO.
    data_start = 1 + (1 if include_references else 0) + (1 if include_list else 0)

    # AtomAij.fits.
    aij_entries = [(aij_extname(ion.ion_id, tag), i + data_start) for i, ion in enumerate(ions)]
    aij_refs = [(aij_extname(ion.ion_id, tag), ion.ref_aij) for ion in ions]
    aij_hdus = [primary_hdu()]
    if include_references:
        aij_hdus.append(make_references_hdu(aij_refs))
    if include_list:
        aij_hdus.append(make_list_hdu(aij_entries, "AIJ_DATA"))
    aij_hdus.extend(make_aij_hdu(ion, tag) for ion in ions)
    fits.HDUList(aij_hdus).writeto(out_dir / "AtomAij.fits", overwrite=overwrite, checksum=True)

    # AtomElj.fits.
    elj_entries = [(elj_extname(ion.ion_id), i + data_start) for i, ion in enumerate(ions)]
    elj_refs = [(elj_extname(ion.ion_id), ion.ref_elj) for ion in ions]
    elj_hdus = [primary_hdu()]
    if include_references:
        elj_hdus.append(make_references_hdu(elj_refs))
    if include_list:
        elj_hdus.append(make_list_hdu(elj_entries, "EJ_DATA"))
    elj_hdus.extend(make_elj_hdu(ion) for ion in ions)
    fits.HDUList(elj_hdus).writeto(out_dir / "AtomElj.fits", overwrite=overwrite, checksum=True)

    # AtomOmij.fits. Ions without electron collision data are excluded from References/data.
    omij_ions = [ion for ion in ions if ion.splups]
    omij_entries = [(omij_extname(ion.ion_id, tag), i + data_start) for i, ion in enumerate(omij_ions)]
    omij_refs = [(omij_extname(ion.ion_id, tag), ion.ref_omij) for ion in omij_ions]
    omij_hdus = [primary_hdu()]
    if include_references:
        omij_hdus.append(make_references_hdu(omij_refs))
    if include_list:
        omij_hdus.append(make_list_hdu(omij_entries, "OMIJ_DATA"))
    omij_hdus.extend(make_omij_hdu(ion, tag, temperatures) for ion in omij_ions)
    fits.HDUList(omij_hdus).writeto(out_dir / "AtomOmij.fits", overwrite=overwrite, checksum=True)

    skipped = [ion.ion_id for ion in ions if not ion.splups]
    print(f"Wrote {out_dir / 'AtomAij.fits'}")
    print(f"Wrote {out_dir / 'AtomElj.fits'}")
    print(f"Wrote {out_dir / 'AtomOmij.fits'}")
    if skipped:
        print("WARNING: no .splups/.scups electron-collision data for:", ", ".join(skipped))


def parse_temperature_grid(args) -> np.ndarray:
    if args.temp_grid:
        vals = [float(tok) for tok in re.split(r"[,\s]+", args.temp_grid.strip()) if tok]
        if len(vals) < 2:
            raise ValueError("--temp-grid must contain at least two temperatures in K")
        T = np.array(vals, dtype=np.float64)
    else:
        T = np.logspace(args.temp_log10_min, args.temp_log10_max, args.temp_points)
    T = np.unique(T[np.isfinite(T) & (T > 0.0)])
    if T.size < 2:
        raise ValueError("Temperature grid must contain at least two positive finite values")
    return T


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build AtomNeb-style FITS files from a CHIANTI ASCII tree.")
    ap.add_argument("--chianti-root", required=True, help="Root of the CHIANTI ASCII tree, e.g. ./chianti")
    ap.add_argument("--out-dir", required=True, help="Directory where AtomAij/AtomElj/AtomOmij.fits will be written")
    ap.add_argument("--tag", default="CHI_CUSTOM", help="Dataset tag appended to Aij/Omij extension names, e.g. CHI90 or CHI_CUSTOM")
    ap.add_argument("--ions", default="", help="Comma/space-separated ion ids to convert, e.g. 'o_3 ne_9 fe_17'. Default: all found ions.")
    ap.add_argument("--max-levels", type=int, default=0, help="Optional maximum level number. Default 0 means all levels available.")
    ap.add_argument(
        "--collision-format",
        choices=["auto", "splups", "scups"],
        default="auto",
        help="Electron collision-strength file format. auto prefers CHIANTI 11 .scups when present, otherwise .splups.",
    )
    ap.add_argument("--temp-grid", default="", help="Explicit temperature grid in K, comma/space-separated. Overrides log grid options.")
    ap.add_argument("--temp-log10-min", type=float, default=2.0)
    ap.add_argument("--temp-log10-max", type=float, default=9.0)
    ap.add_argument("--temp-points", type=int, default=71)
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing FITS files")
    ap.add_argument(
        "--include-list",
        action="store_true",
        help="Add a nonstandard List HDU after References. Default matches AtomNeb chianti90: no List HDU.",
    )
    ap.add_argument(
        "--no-references",
        action="store_true",
        help=(
            "Do not include the References HDU in AtomAij.fits, AtomElj.fits, "
            "or AtomOmij.fits. By default, References is included to match "
            "AtomNeb-style FITS files."
        ),
    )
    args = ap.parse_args(argv)

    root = Path(args.chianti_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    selected = [x for x in re.split(r"[,\s]+", args.ions.strip()) if x] or None
    max_levels = args.max_levels if args.max_levels and args.max_levels > 0 else None
    temperatures = parse_temperature_grid(args)

    ion_dirs = find_ion_dirs(root, selected_ions=selected)
    if not ion_dirs:
        raise SystemExit(f"ERROR: no CHIANTI ions with .elvlc and .wgfa found below {root}")

    ions: List[IonData] = []
    for ion_id, ion_dir in ion_dirs:
        try:
            ion = read_ion(ion_id, ion_dir, max_levels=max_levels, collision_format=args.collision_format)
            ions.append(ion)
            print(
                f"{ion_id:8s}: Z={ion_sort_key(ion_id)[0]:2d} "
                f"levels={len(ion.levels):4d} Aij={np.count_nonzero(ion.aij):6d} "
                f"collisions={len(ion.splups):6d} "
                f"format={(ion.splups[0].source_format if ion.splups else 'none')}"
            )
        except Exception as exc:
            print(f"WARNING: skipping {ion_id} at {ion_dir}: {exc}")

    if not ions:
        raise SystemExit("ERROR: no ions could be parsed successfully")

    write_fits_files(ions, out_dir, args.tag, temperatures, overwrite=not args.no_overwrite, include_list=args.include_list, include_references=not args.no_references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
