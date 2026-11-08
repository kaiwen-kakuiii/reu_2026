"""Rapid-variability error floor via light-curve median filtering.

MOVED OUT of spec_lc_pipeline.ipynb (2026-06). That pipeline builds the
**WIRO-only** light curves and stops at the per-epoch error budget:

    err = sqrt(err_spec^2 + scatter^2 + syserr^2)

The median-filter term estimates EXCESS point-to-point scatter that the formal
per-epoch errors miss. It is a *temporal* quantity (scatter of the light curve
across epochs), so it only makes sense on the FINAL combined light curve, after
intercalibrating and merging WIRO with the other surveys -- NOT on the WIRO-only
LC. Apply it here, at the multi-survey stage.

Caveat: the median filter treats genuine fast AGN variability as "scatter", so a
too-small window inflates the errors with real signal (and can wash out a real
lag). Tune `size` to the combined cadence.
"""

import numpy as np
from scipy.ndimage import median_filter


def median_filter_scatter(lc, size=5):
    """Std of (lc - median_filtered(lc)): one excess-scatter scalar.

    Parameters
    ----------
    lc : array_like
        Flux light curve, ordered by time.
    size : int
        Median-filter window in epochs. Larger -> absorbs less real variability
        into the floor.

    Returns
    -------
    float
        A single scalar to add IN QUADRATURE to every epoch's error.
    """
    lc = np.asarray(lc, dtype=float)
    smooth = median_filter(lc, size=size, mode="nearest")
    return np.std(lc - smooth)


def add_variability_floor(flux, err, size=5):
    """Return `err` with the median-filter excess-scatter floor folded in:

        err_out = sqrt(err^2 + median_filter_scatter(flux, size)^2)

    Use on the COMBINED multi-survey light curve.
    """
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    floor = median_filter_scatter(flux, size=size)
    return np.sqrt(err ** 2 + floor ** 2)


def _cli(argv=None):
    """CLI: fold the median-filter floor into a light curve's error column.

        python median_filter_floor.py <lc> [-o OUT] [--size N] [--no-sort]

    Input  : whitespace LC with columns  JD  flux  err_ccf  [extra...].
    Output : same file with the CCF error column replaced by
                 sqrt(err^2 + floor^2)
             If the header declares 'columns: JD flux err_ccf err_total',
             add the same floor in quadrature to err_total, preserving the
             shared narrow variance err_total^2 - err_ccf^2. Other extra
             columns (e.g. a survey label) are preserved verbatim.
    The floor is a single scalar (one number for the whole LC). Because it is
    temporal, the LC is sorted by JD to compute it (disable with --no-sort);
    rows are written back in their ORIGINAL order.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Add the median-filter rapid-variability error floor to a light curve.")
    ap.add_argument("lc", help="input light curve: columns 'JD flux err_ccf [extra...]'")
    ap.add_argument("-o", "--out", help="output path (default: <stem>_mffloor<ext>)")
    ap.add_argument("--size", type=int, default=5,
                    help="median-filter window in epochs (default 5)")
    ap.add_argument("--no-sort", action="store_true",
                    help="compute the floor in file order instead of sorting by JD")
    args = ap.parse_args(argv)

    with open(args.lc) as f:
        input_lines = f.readlines()
    has_total_error = any(
        line.startswith("#") and line.lstrip("# ").split()[:5]
        == ["columns:", "JD", "flux", "err_ccf", "err_total"]
        for line in input_lines)
    jd, flux, err = np.loadtxt(args.lc, usecols=(0, 1, 2), unpack=True)

    if args.no_sort:
        floor = median_filter_scatter(flux, size=args.size)
    else:
        floor = median_filter_scatter(flux[np.argsort(jd)], size=args.size)
    err_new = np.sqrt(err ** 2 + floor ** 2)
    if has_total_error:
        err_total = np.loadtxt(args.lc, usecols=3)
        err_total_new = np.sqrt(err_total ** 2 + floor ** 2)

    out = args.out
    if out is None:
        root, ext = os.path.splitext(args.lc)
        out = f"{root}_mffloor{ext or '.txt'}"

    # Rewrite line-by-line so headers and unrelated extra columns survive.
    # loadtxt skips comments/blanks, so the data-row index k stays aligned.
    out_lines, k = [], 0
    for line in input_lines:
        s = line.rstrip("\n")
        t = s.strip()
        if not t or t.startswith("#"):
            out_lines.append(s)
            continue
        toks = t.split()
        toks[2] = f"{err_new[k]:.18e}"
        if has_total_error:
            toks[3] = f"{err_total_new[k]:.18e}"
        out_lines.append(" ".join(toks))
        k += 1
    with open(out, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    frac = np.median(floor / np.abs(flux))
    print(f"median-filter floor = {floor:.4e}  (size={args.size}, ~{frac:.2%} of |flux|)")
    print(f"err median: {np.median(err):.4e} -> {np.median(err_new):.4e}")
    if has_total_error:
        print("err_total updated; shared narrow-model variance preserved")
    print(f"wrote {out}  ({k} epochs)")


if __name__ == "__main__":
    _cli()
