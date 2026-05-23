# chianti2atomneb

`chianti2atomneb.py` converts raw CHIANTI atomic-data folders into
AtomNeb-style FITS files.

It is intended for workflows that want to use CHIANTI atomic data through
AtomNeb/pyEQUIB-style FITS tables rather than reading the original CHIANTI
ASCII files directly.

The converter writes:

```text
AtomAij.fits   # radiative transition probabilities
AtomElj.fits   # energy levels
AtomOmij.fits  # electron effective collision strengths on a temperature grid
```

## Relationship to `chianti-raw`

`chianti2atomneb.py` is intentionally kept separate from the lightweight
`chianti-raw` package.

- `chianti-raw` reads CHIANTI files directly and only needs NumPy.
- `chianti2atomneb.py` writes FITS files and therefore requires Astropy.

This keeps the direct reader lightweight while still providing a conversion
tool when AtomNeb-compatible FITS files are needed.

## Supported CHIANTI data layouts

The converter supports several CHIANTI generations and Cloudy-converted layouts:

| Source | Level file | A-value file | Collision file |
|---|---|---|---|
| CHIANTI 6/7 | `.elvlc` | `.wgfa` | `.splups` |
| original CHIANTI 8/9/10/11 | `.elvlc` | `.wgfa` | `.scups` |
| Cloudy-converted CHIANTI 9-style | `.elvlc` | `.wgfa` | `.splups` |

The standard CHIANTI tree layout is preferred:

```text
chianti_root/
  o/
    o_3/
      o_3.elvlc
      o_3.wgfa
      o_3.scups
```

The converter also accepts compact single-ion layouts such as:

```text
chianti_root/
  o/
    o_3.elvlc
    o_3.wgfa
    o_3.scups
```

or:

```text
chianti_root/
  o_3.elvlc
  o_3.wgfa
  o_3.scups
```

## Requirements

```bash
pip install numpy astropy
```

No ChiantiPy installation is required.

## Basic usage

Convert a full CHIANTI tree:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110 \
  --tag CHI110
```

This creates:

```text
./atomneb_chianti110/AtomAij.fits
./atomneb_chianti110/AtomElj.fits
./atomneb_chianti110/AtomOmij.fits
```

## Convert only selected ions

Use CHIANTI ion folder names:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --ions o_3,ne_3,mg_5 \
  --out-dir ./atomneb_subset \
  --tag CHI110
```

## Collision file format

The converter can auto-detect `.scups` or `.splups`:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110 \
  --tag CHI110 \
  --include-list \
  --collision-format auto
```

Available modes:

```text
auto    prefer .scups if present, otherwise use .splups
scups   force original CHIANTI .scups format
splups  force older/Cloudy .splups format
```

Use `scups` for original CHIANTI 8/9/10/11:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti10 \
  --out-dir ./atomneb_chianti100 \
  --tag CHI100 \
  --include-list \
  --collision-format scups
```

Use `splups` for CHIANTI 6/7 or Cloudy-converted CHIANTI 9-style files:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti7 \
  --out-dir ./atomneb_chianti70 \
  --tag CHI70 \
  --include-list \
  --collision-format splups
```

## Temperature grid for `AtomOmij.fits`

CHIANTI `.scups` and `.splups` store scaled effective collision strengths.
`AtomOmij.fits` stores values evaluated on a physical temperature grid.

By default, the converter uses a logarithmic grid. You can control it with:

```bash
--temp-log10-min 2
--temp-log10-max 9
--temp-points 71
```

Example:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110 \
  --tag CHI110 \
  --include-list \
  --temp-log10-min 2 \
  --temp-log10-max 9 \
  --temp-points 71
```

This produces:

```text
log10(T/K) = 2.0, 2.1, 2.2, ..., 9.0
```

For a denser grid:

```bash
--temp-log10-min 3 --temp-log10-max 9 --temp-points 121
```

## Limiting the number of levels

To restrict each ion to the first `N` levels:

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110_70lev \
  --tag CHI110 \
  --include-list \
  --max-levels 70
```

This also filters A-values and collision records to transitions whose upper and
lower levels are within the retained level range.

## FITS HDU layout

Default layout:

```text
PRIMARY
References
ion data extensions...
```

With `--include-list`:

```text
PRIMARY
References
List
ion data extensions...
```

With `--no-references`:

```text
PRIMARY
ion data extensions...
```

With both `--include-list --no-references`:

```text
PRIMARY
List
ion data extensions...
```

## Extension naming

The converter writes AtomNeb-style extension names using lowercase ion labels
and the supplied tag.

Examples with `--tag CHI110`:

```text
AtomAij.fits:
  o_iii_aij_CHI110

AtomElj.fits:
  o_iii_elj

AtomOmij.fits:
  o_iii_omij_CHI110
```

The `List` HDU, when requested, stores extension names and HDU numbers.

## References HDU

By default, each output FITS file includes a `References` HDU with columns:

```text
ATOMICDATA
REFERENCE
```

References are extracted from the relevant CHIANTI source files:

```text
AtomElj.fits   <- .elvlc references
AtomAij.fits   <- .wgfa references
AtomOmij.fits  <- .scups or .splups references
```

The converter cleans common CHIANTI comment prefixes such as:

```text
% A values:
% oscillator strengths:
% Effective collision strength:
Excitation data from:
```

Use `--no-references` to omit the `References` HDU.

## List HDU

Use `--include-list` to include a `List` HDU.

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110 \
  --tag CHI110 \
  --include-list
```

This is useful for inspecting output files with FITS viewers such as `fv`, or
for workflows that expect a list of available ion extensions.

## Recommended cleanup before conversion

For original CHIANTI 8/9/10/11 folders, keep:

```text
.elvlc
.wgfa
.scups
```

For CHIANTI 6/7 and Cloudy-converted CHIANTI 9-style folders, keep:

```text
.elvlc
.wgfa
.splups
```

A safe rule for mixed-version archives is to keep:

```text
.elvlc
.wgfa
.scups
.splups
```

Proton collision files such as `.psplups` are not used by this converter.

## Example: CHIANTI 11 to AtomNeb-style FITS

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110 \
  --tag CHI110 \
  --collision-format auto \
  --temp-log10-min 2 \
  --temp-log10-max 9 \
  --temp-points 71 \
  --include-list
```

## Example: original CHIANTI 9

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti9_original \
  --out-dir ./atomneb_chianti90_original \
  --tag CHI90 \
  --collision-format scups
  --include-list
```

## Example: Cloudy-converted CHIANTI 9-style data

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti9_cloudy \
  --out-dir ./atomneb_chianti90_cloudy \
  --tag CHI90CLOUDY \
  --collision-format splups
  --include-list
```

## Example: no References HDU

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110_no_refs \
  --tag CHI110 \
  --no-references
```

## Example: include `List` but omit `References`

```bash
python chianti2atomneb.py \
  --chianti-root ./chianti11 \
  --out-dir ./atomneb_chianti110_list_only \
  --tag CHI110 \
  --include-list \
  --no-references
```

## Validation checks

After conversion, inspect the output with Python:

```python
from astropy.io import fits

for fname in ["AtomElj.fits", "AtomAij.fits", "AtomOmij.fits"]:
    hdul = fits.open(f"./atomneb_chianti110/{fname}")
    print(fname)
    for i, hdu in enumerate(hdul):
        print(i, hdu.name)
```

Check one O III level table:

```python
from astropy.io import fits

with fits.open("./atomneb_chianti110/AtomElj.fits") as hdul:
    data = hdul["o_iii_elj"].data
    print(data.names)
    print(data[:5])
```

Check one O III collision table:

```python
from astropy.io import fits

with fits.open("./atomneb_chianti110/AtomOmij.fits") as hdul:
    data = hdul["o_iii_omij_CHI110"].data
    print(data.names)
    print(data[0])
```

## Notes and limitations

- `AtomOmij.fits` stores effective collision strengths evaluated on a chosen
  temperature grid, not the original CHIANTI scaled spline coefficients.
- The converter currently handles electron-impact collision strengths only.
- Proton collision data such as `.psplups` are ignored.
- The converter is independent of ChiantiPy.
- The converter is independent of the `chianti-raw` package, although both use
  the same parsing philosophy.

## Main command-line options

The detected options in the current script are:

- `--chianti-root`
- `--collision-format`
- `--include-list`
- `--ions`
- `--max-levels`
- `--no-overwrite`
- `--no-references`
- `--out-dir`
- `--tag`
- `--temp-grid`
- `--temp-log10-max`
- `--temp-log10-min`
- `--temp-points`

For authoritative help:

```bash
python chianti2atomneb.py --help
```
