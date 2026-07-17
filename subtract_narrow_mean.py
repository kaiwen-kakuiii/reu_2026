#!/usr/bin/env python3
"""
Narrow-line subtraction for a campaign mean spectrum (CLI version).

Same pipeline as subtract_narrow_mean.ipynb — keep the two in sync.
The notebook is for interactive verification; this script is for mass
execution over many objects.

Input spectrum is in REST-FRAME wavelength (3 columns: wave flux err).
All windows are rest-frame Angstrom, given on the command line.

This script runs FIRST in the workflow. With --input-dir/--flux-lst it also
runs step 0 (moved from spec_lc_pipeline): stack the AGN-only rest-frame
per-epoch combined spectra into the campaign mean & RMS spectra, write
spec_mean_<line>.txt / spec_rms_<line>.txt, and fit the narrow model on that
mean (the positional spec argument is then ignored). spec_lc_pipeline and
measure_linewidth load these files back -- they no longer build them.

Pipeline (logic of code_asymmetry.py:subtract_narrow_lines, with the [OIII]
steps decoupled from the global continuum):
  0. (optional) stack per-epoch combined spectra -> mean & RMS spectra
     (saved); the mean becomes the fit input
  1. subtract [OIII]4959 from the RAW flux using the [OIII]5007 profile as
     template (free: wavelength shift, flux ratio, local linear background;
     the background absorbs the continuum under 4959)
  2. extract the clean [OIII]5007 profile with its own local linear
     continuum -> this profile is the narrow-line template
  3. global linear continuum from two line-free windows (median flux) is
     subtracted ONLY here: fit the Hbeta window with (shifted/scaled
     template) + two Gaussians + constant, then subtract only the narrow
     template part

Uncertainties (Monte Carlo): flux perturbed by err * N(0,1), N re-runs;
per-pixel scatter of the narrow model gives its 1-sigma uncertainty; the
error written for the subtracted spectrum is sqrt(err^2 + sigma_narrow^2).

Outputs (obj tag = --obj, default: current directory name)
  spec_mean_<line>.txt            campaign mean spectrum (with --input-dir)
  spec_rms_<line>.txt             campaign RMS spectrum  (with --input-dir)
  <spec>.subnarrow                wave  flux_sub  err_combined
  <obj>_narrow_profile.txt        narrow model and components +/- 1 sigma
  <obj>_oiii5007_template.txt     clean template: wave flux err sigma_mc
  <obj>_subtract_narrow_mean.pdf  four-panel diagnostic figure

Usage
-----
  # step 0 + narrow fit (build the mean/rms, then fit the mean):
  python subtract_narrow_mean.py \\
      --input-dir agn_only_dered_spectra --flux-lst flux.lst \\
      --contil 4670 4740 --contir 5080 5150 --oiii 4990 5022 \\
      [--line-name hbeta] [--exclude rebin .out] \\
      [--oiii4959 4935 4977] [--hb 4800 4920] \\
      [--obj NAME] [--mc 500] [--seed 42] [--no-plot] [--show]

  # narrow fit only, on an existing rest-frame mean spectrum:
  python subtract_narrow_mean.py spec_mean_hbeta.txt \\
      --contil 4670 4740 --contir 5080 5150 --oiii 4990 5022 [...]
"""

import argparse
import os

import numpy as np
from lmfit import minimize, Parameters

UNIT = 1.0e-14

# Okabe-Ito colorblind-safe palette, fixed assignment
C_DATA = '#000000'       # observed spectrum
C_MODEL = '#E69F00'      # total model
C_BROAD = '#0072B2'      # broad Hbeta
C_NARROW = '#009E73'     # narrow components / template
C_SUB = '#D55E00'        # narrow-subtracted spectrum
C_CON = '#CC79A7'        # continuum / background lines


def read_spec(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]


# ---------------- step 0: campaign mean & RMS spectra ----------------
# Moved from spec_lc_pipeline: this script runs FIRST in the workflow, so the
# campaign mean (narrow-fit input) and RMS are built here; spec_lc_pipeline
# and measure_linewidth load the saved files back.

def load_spectrum(path):
    """Return (wavelength, flux, err) sorted by wavelength.

    Robust to either (N, 3) column layout or (3, N) row layout.
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2:
        raise ValueError('Spectrum %s is not 2-D (shape %s).' % (path, arr.shape))
    if arr.shape[1] == 3:        # (N, 3): columns are wl, flux, err
        lam, f, e = arr.T
    elif arr.shape[0] == 3:      # (3, N): rows are wl, flux, err
        lam, f, e = arr
    else:
        raise ValueError('Spectrum %s is not 3-column (shape %s).'
                         % (path, arr.shape))
    order = np.argsort(lam)
    return lam[order], f[order], e[order]


def read_epochs(flux_lst, input_dir, combined_pattern, exclude):
    """(name, jd) of the combined epochs listed in flux_lst, sorted by JD."""
    names, jds = [], []
    with open(flux_lst) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            if combined_pattern not in name or \
                    any(s in name for s in exclude):
                continue
            try:
                jd = float(parts[1])
            except ValueError:          # header / non-numeric line
                continue
            if not os.path.exists(os.path.join(input_dir, name)):
                print('  [skip] %s: not found in input dir' % name)
                continue
            names.append(name)
            jds.append(jd)
    order = np.argsort(jds)
    return [names[i] for i in order], np.asarray(jds, float)[order]


def build_mean_rms(input_dir, flux_lst, line_name, combined_pattern, exclude):
    """Stack the per-epoch combined spectra; write mean & RMS spectra.

    Returns the path of the mean spectrum (the narrow-fit input). Files are
    3-column (wave, flux, err) in RAW flux units so downstream readers
    (spec_lc_pipeline, measure_linewidth) load them unchanged.
    """
    epoch_names, jd = read_epochs(flux_lst, input_dir, combined_pattern,
                                  exclude)
    if not epoch_names:
        raise ValueError('no combined epochs found in %s' % flux_lst)
    print('%d epochs | JD %.2f -> %.2f' % (len(epoch_names), jd.min(),
                                           jd.max()))

    wl_grid, stack_f, stack_e = None, [], []
    for name in epoch_names:
        lam, f, e = load_spectrum(os.path.join(input_dir, name))
        if wl_grid is None:
            wl_grid = lam
        elif len(lam) != len(wl_grid) or not np.allclose(lam, wl_grid):
            raise ValueError('%s is on a different wavelength grid; '
                             'resample onto a common grid first.' % name)
        stack_f.append(f)
        stack_e.append(e)
    stack_f = np.array(stack_f)
    stack_e = np.array(stack_e)

    mean_flux = stack_f.mean(axis=0)
    # propagated error of the pixel-wise mean: sqrt(sum err_i^2) / N
    mean_err = np.sqrt(np.sum(stack_e ** 2, axis=0)) / stack_e.shape[0]
    rms_flux = stack_f.std(axis=0)      # population std (ddof=0) across epochs
    # error of the std itself: rms / sqrt(2(N-1))  (Gaussian approximation)
    rms_err = rms_flux / np.sqrt(2.0 * (stack_f.shape[0] - 1))
    print('Stacked %d spectra x %d pixels' % stack_f.shape)
    print('median S/N of mean spectrum: %.1f'
          % np.median(mean_flux / mean_err))

    mean_out = 'spec_mean_%s.txt' % line_name
    rms_out = 'spec_rms_%s.txt' % line_name
    np.savetxt(mean_out, np.vstack((wl_grid, mean_flux, mean_err)).T)
    np.savetxt(rms_out, np.vstack((wl_grid, rms_flux, rms_err)).T)
    print('mean spectrum          : %s' % mean_out)
    print('rms spectrum           : %s' % rms_out)
    return mean_out


def window_median(wave, flux, lim1, lim2):
    """Median flux and mean wavelength inside [lim1, lim2]."""
    idx = np.where((wave >= lim1) & (wave <= lim2))[0]
    if idx.size == 0:
        raise ValueError('empty window %.1f-%.1f' % (lim1, lim2))
    return np.median(flux[idx]), np.mean(wave[idx])


def linear_through(w1, f1, w2, f2, wave):
    return f1 + (f2 - f1) / (w2 - w1) * (wave - w1)


def gauss(w, area, center, sigma):
    """Gaussian with unit area scaled by `area`."""
    return area / np.sqrt(2.0 * np.pi) / sigma * \
        np.exp(-(w - center)**2 / 2.0 / sigma**2)


def run_pipeline(wave, flux, err, p):
    """One full narrow-line subtraction pass. Returns a result dict.

    p: dict with keys conti_left, conti_right, oiii_win, oiii4959_win,
    hb_win — each a (lo, hi) rest-frame window.

    The [OIII] steps (1-2) run on the RAW flux: each carries its own local
    linear continuum/background, so the global continuum plays no role there.
    The global linear continuum is subtracted only for the Hbeta fit (step 3),
    whose broad-line background model is just a constant.
    """
    # ---- step 1: subtract [OIII]4959 using the 5007 profile ------------
    o1, o2 = p['oiii_win']
    idx = np.where((wave >= o1) & (wave <= o2))[0]
    wave_o1s, flux_o1s = wave[idx], flux[idx]
    fl, wl = window_median(wave, flux, o1 - 2, o1)
    fr, wr = window_median(wave, flux, o2, o2 + 3)
    fcon_o1s = linear_through(wl, fl, wr, fr, wave_o1s)
    flux_o1s = flux_o1s - fcon_o1s

    def model_4959(params, w, with_background=True):
        m = np.interp(w, wave_o1s - params['oiii_diff'],
                      flux_o1s / params['oiii_ratio'], left=0.0, right=0.0)
        if with_background:
            m = m + params['a_bak'] + params['b_bak'] * w
        return m

    def resi_4959(params, w, f, e):
        return (f - model_4959(params, w)) / e

    pars1 = Parameters()
    pars1.add('oiii_diff', value=5007.0 - 4959.0)
    pars1.add('a_bak', value=0.0)
    pars1.add('b_bak', value=0.0)
    pars1.add('oiii_ratio', value=3.0)

    idx49 = np.where((wave >= p['oiii4959_win'][0]) &
                     (wave <= p['oiii4959_win'][1]))[0]
    out1 = minimize(resi_4959, pars1,
                    args=(wave[idx49], flux[idx49], err[idx49]))

    narrow_4959 = model_4959(out1.params, wave, with_background=False)
    flux1 = flux - narrow_4959

    # ---- step 2: clean [OIII]5007 profile (the narrow template) --------
    fl, wl = window_median(wave, flux1, o1 - 2, o1 + 3)
    fr, wr = window_median(wave, flux1, o2 - 2, o2 + 3)
    wave_t = wave[idx]
    fcon_5007 = linear_through(wl, fl, wr, fr, wave_t)
    flux_t = flux1[idx] - fcon_5007
    err_t = err[idx]

    narrow_5007 = np.interp(wave, wave_t, flux_t, left=0.0, right=0.0)

    # ---- step 3: Hbeta = narrow template + 2 Gaussians + constant ------
    # Global linear continuum subtracted here only, and measured on the
    # NARROW-SUBTRACTED flux: continuum windows may sit near [OIII]
    # (broad Hbeta extends underneath), and the fitted narrow lines must
    # not be counted in the continuum level (avoids double subtraction).
    flux_ns = flux1 - narrow_5007
    fl, wl = window_median(wave, flux_ns, *p['conti_left'])
    fr, wr = window_median(wave, flux_ns, *p['conti_right'])
    fcon_tot = linear_through(wl, fl, wr, fr, wave)
    fluxc = flux_ns - fcon_tot

    hb1, hb2 = p['hb_win']
    idxhb = np.where((wave >= hb1) & (wave <= hb2))[0]
    wave_hb, flux_hb, err_hb = wave[idxhb], fluxc[idxhb], err[idxhb]

    def hb_narrow(params, w):
        return np.interp(w, wave_t - params['shift'],
                         flux_t / params['ratio'], left=0.0, right=0.0)

    def hb_broad(params, w):
        g1 = gauss(w, params['p0'], params['p1'], params['p2'])
        g2 = gauss(w, params['p3'], params['p4'], params['p5'])
        return g1 + g2 + params['p6']

    def resi_hb(params, w, f, e):
        return (f - hb_narrow(params, w) - hb_broad(params, w)) / e

    # Amplitude guesses from the integrated window flux, centers/shift
    # bounded: fixed small guesses can fall into a local minimum with a
    # Gaussian far outside the window.
    dlam = np.median(np.diff(wave_hb))
    area0 = max(np.sum(flux_hb) * dlam, 10.0 * dlam)
    shift0 = 5007.0 - 4861.0

    pars3 = Parameters()
    pars3.add('shift', value=shift0, min=shift0 - 20.0, max=shift0 + 20.0)
    pars3.add('ratio', value=10.0, min=1.0, max=15.0)
    pars3.add('p0', value=0.3 * area0, min=0.0)
    pars3.add('p1', value=4861.0, min=hb1, max=hb2)
    pars3.add('p2', value=2000.0 / 3.0e5 * 4861.0,
              min=1000.0 / 3.0e5 * 4861.0)
    pars3.add('p3', value=0.5 * area0, min=0.0)
    pars3.add('p4', value=4861.0, min=hb1, max=hb2)
    pars3.add('p5', value=4000.0 / 3.0e5 * 4861.0,
              min=1000.0 / 3.0e5 * 4861.0)
    pars3.add('p6', value=0.0)

    out3 = minimize(resi_hb, pars3, args=(wave_hb, flux_hb, err_hb))

    narrow_hb = np.interp(wave, wave_hb, hb_narrow(out3.params, wave_hb),
                          left=0.0, right=0.0)

    narrow_total = narrow_4959 + narrow_5007 + narrow_hb
    flux_sub = flux - narrow_total   # broad Hbeta + continuum kept

    return {
        'flux_sub': flux_sub,
        'narrow_total': narrow_total,
        'narrow_4959': narrow_4959,
        'narrow_5007': narrow_5007,
        'narrow_hb': narrow_hb,
        'fcon_tot': fcon_tot,
        'wave_t': wave_t, 'flux_t': flux_t, 'err_t': err_t,
        'fcon_5007': fcon_5007,
        'wave_o1s': wave_o1s, 'flux_o1s': flux_o1s, 'fcon_o1s': fcon_o1s,
        'wave_hb': wave_hb, 'flux_hb': flux_hb,
        'hb_narrow_model': hb_narrow(out3.params, wave_hb),
        'hb_broad_model': hb_broad(out3.params, wave_hb),
        'out1': out1, 'out3': out3,
    }


def monte_carlo(wave, flux, err, p, n, seed):
    """Re-run the pipeline on n flux realizations perturbed by err."""
    rng = np.random.default_rng(seed)
    keys = ('narrow_total', 'narrow_4959', 'narrow_5007', 'narrow_hb',
            'flux_t')
    stacks = {k: [] for k in keys}
    samples = {k: [] for k in ('oiii_ratio', 'oiii_diff', 'ratio', 'shift')}
    nfail = 0
    for _ in range(n):
        f_pert = flux + err * rng.standard_normal(flux.size)
        try:
            r = run_pipeline(wave, f_pert, err, p)
        except Exception:
            nfail += 1
            continue
        for k in keys:
            stacks[k].append(r[k])
        samples['oiii_ratio'].append(r['out1'].params['oiii_ratio'].value)
        samples['oiii_diff'].append(r['out1'].params['oiii_diff'].value)
        samples['ratio'].append(r['out3'].params['ratio'].value)
        samples['shift'].append(r['out3'].params['shift'].value)

    result = {'nfail': nfail, 'nok': n - nfail}
    for k in keys:
        result['sig_' + k] = np.std(np.array(stacks[k]), axis=0, ddof=1)
    for k, v in samples.items():
        arr = np.array(v)
        result[k] = (np.median(arr), np.std(arr, ddof=1))
    return result


def fmt_par(par):
    if par.stderr is None:
        return '%.4f +/- n/a' % par.value
    return '%.4f +/- %.4f' % (par.value, par.stderr)


def make_plot(wave, flux, err, p, r, mc, obj_name, outfile, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(obj_name)
    for ax in axes.flat:
        ax.grid(alpha=0.2, linewidth=0.5)

    # panel 1: [OIII]4959 subtraction (raw-flux space)
    ax = axes[0, 0]
    idx = np.where((wave >= 4650.0) & (wave <= 5200.0))[0]
    ax.plot(wave[idx], flux[idx], color=C_DATA, lw=1.0, label='obs')
    ax.plot(r['wave_o1s'], r['flux_o1s'] + r['fcon_o1s'], ':',
            color=C_NARROW, label='[OIII]5007')
    ax.plot(r['wave_o1s'], r['fcon_o1s'], ':', color=C_CON,
            label='5007 local con')
    ax.plot(wave[idx], (flux - r['narrow_4959'])[idx], color=C_SUB, lw=1.0,
            label='4959 subtracted')
    ax.plot(wave[idx], r['narrow_4959'][idx], '--', color=C_MODEL,
            label='4959 model')
    ax.set_title('[OIII]4959 subtraction')
    ax.legend(fontsize=8)

    # panel 2: [OIII]5007 template extraction (raw-flux space)
    ax = axes[0, 1]
    o1, o2 = p['oiii_win']
    idx = np.where((wave >= o1 - 50) & (wave <= o2 + 80))[0]
    ax.plot(wave[idx], (flux - r['narrow_4959'])[idx], color=C_DATA, lw=1.0,
            label='4959 subtracted')
    ax.plot(r['wave_t'], r['fcon_5007'], '--', color=C_CON, label='local con')
    ax.plot(r['wave_t'], r['flux_t'], color=C_NARROW, label='5007 template')
    if mc is not None:
        ax.fill_between(r['wave_t'], r['flux_t'] - mc['sig_flux_t'],
                        r['flux_t'] + mc['sig_flux_t'], color=C_NARROW,
                        alpha=0.25, linewidth=0, label='template 1$\\sigma$')
    ax.set_title('[OIII]5007 template')
    ax.legend(fontsize=8)

    # panel 3: Hbeta fit
    ax = axes[1, 0]
    ax.plot(r['wave_hb'], r['flux_hb'], color=C_DATA, lw=1.0, label='data')
    ax.plot(r['wave_hb'], r['hb_narrow_model'] + r['hb_broad_model'], '--',
            color=C_MODEL, label='total model')
    ax.plot(r['wave_hb'], r['hb_broad_model'], '--', color=C_BROAD,
            label='broad')
    ax.plot(r['wave_hb'], r['hb_narrow_model'], '--', color=C_NARROW,
            label='narrow Hbeta')
    ax.set_title('Hbeta fit')
    ax.legend(fontsize=8)

    # panel 4: final result
    ax = axes[1, 1]
    idx = np.where((wave >= 4700.0) & (wave <= 5150.0))[0]
    ax.plot(wave[idx], flux[idx], color=C_DATA, lw=1.0, label='obs')
    ax.plot(wave[idx], r['flux_sub'][idx], color=C_SUB, lw=1.0,
            label='narrow subtracted')
    if mc is not None:
        ax.fill_between(wave[idx],
                        (r['flux_sub'] - mc['sig_narrow_total'])[idx],
                        (r['flux_sub'] + mc['sig_narrow_total'])[idx],
                        color=C_SUB, alpha=0.25, linewidth=0,
                        label='narrow model 1$\\sigma$')
    ax.plot(wave[idx], r['narrow_total'][idx], '--', color=C_NARROW,
            label='narrow model')
    ax.set_title('final result')
    ax.text(0.95, 0.95, 'Hb narrow/[OIII] ratio: %.3f'
            % r['out3'].params['ratio'].value,
            ha='right', va='top', transform=ax.transAxes, fontsize=9)
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel('rest wavelength ($\\AA$)')
        ax.set_ylabel('flux (%.0e erg/s/cm$^2$/$\\AA$)' % UNIT)
    fig.tight_layout()
    fig.savefig(outfile, format='pdf', bbox_inches='tight')
    print('diagnostic plot: %s' % outfile)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Narrow-line subtraction for a rest-frame mean spectrum '
                    '(logic of code_asymmetry.py, + Monte Carlo errors).')
    ap.add_argument('spec', nargs='?', default='./combined.txt.meanrms',
                    help='spectrum file: wave flux err, rest-frame '
                         '(ignored when --input-dir is given)')
    ap.add_argument('--input-dir', default=None,
                    help='step 0: directory of AGN-only REST-FRAME per-epoch '
                         'combined spectra; build mean & RMS spectra first '
                         'and fit the mean (requires --flux-lst)')
    ap.add_argument('--flux-lst', default=None,
                    help='step 0: epoch list, one "<combined-name> <JD>" per '
                         'row')
    ap.add_argument('--line-name', default='hbeta',
                    help='step 0: tag for spec_mean_/spec_rms_ file names '
                         '(default hbeta)')
    ap.add_argument('--combined-pattern', default='_combined.txt',
                    help='step 0: substring marking a per-epoch combined '
                         'spectrum (default _combined.txt)')
    ap.add_argument('--exclude', nargs='*', default=['rebin', '.out'],
                    metavar='SUBSTR',
                    help='step 0: skip spectra whose name contains any of '
                         'these substrings (default: rebin .out)')
    ap.add_argument('--contil', nargs=2, type=float, required=True,
                    metavar=('LO', 'HI'), help='left global continuum window')
    ap.add_argument('--contir', nargs=2, type=float, required=True,
                    metavar=('LO', 'HI'), help='right global continuum window')
    ap.add_argument('--oiii', nargs=2, type=float, required=True,
                    metavar=('LO', 'HI'),
                    help='[OIII]5007 window (template extent)')
    ap.add_argument('--oiii4959', nargs=2, type=float,
                    default=[4935.0, 4977.0], metavar=('LO', 'HI'),
                    help='[OIII]4959 fit window (default 4935 4977)')
    ap.add_argument('--hb', nargs=2, type=float, default=[4800.0, 4920.0],
                    metavar=('LO', 'HI'),
                    help='Hbeta fit window (default 4800 4920)')
    ap.add_argument('--obj', default=None,
                    help='output name tag (default: current directory name)')
    ap.add_argument('--mc', type=int, default=500,
                    help='number of Monte Carlo realizations (0 = off)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--show', action='store_true',
                    help='open the plot window (default: save PDF only)')
    args = ap.parse_args()

    obj_name = args.obj or os.path.basename(os.getcwd())
    p = {'conti_left': tuple(args.contil),
         'conti_right': tuple(args.contir),
         'oiii_win': tuple(args.oiii),
         'oiii4959_win': tuple(args.oiii4959),
         'hb_win': tuple(args.hb)}

    spec = args.spec
    if args.input_dir is not None:
        if args.flux_lst is None:
            ap.error('--input-dir requires --flux-lst')
        print('--- step 0: mean & RMS spectra ---')
        spec = build_mean_rms(args.input_dir, args.flux_lst, args.line_name,
                              args.combined_pattern, args.exclude)

    wave, flux, err = read_spec(spec)
    flux = flux / UNIT
    err = err / UNIT

    r = run_pipeline(wave, flux, err, p)

    print('--- best fit ---')
    print('[OIII] 5007/4959 ratio : %s' % fmt_par(r['out1'].params['oiii_ratio']))
    print('4959 center            : %.2f'
          % (5007.0 - r['out1'].params['oiii_diff'].value))
    print('step-1 red. chi2       : %.3f' % r['out1'].redchi)
    print('[OIII]/Hb narrow ratio : %s' % fmt_par(r['out3'].params['ratio']))
    print('narrow Hb center       : %.2f'
          % (5007.0 - r['out3'].params['shift'].value))
    print('step-3 red. chi2       : %.3f' % r['out3'].redchi)

    mc = None
    if args.mc > 0:
        print('--- Monte Carlo (%d realizations, seed %d) ---'
              % (args.mc, args.seed))
        mc = monte_carlo(wave, flux, err, p, args.mc, args.seed)
        if mc['nfail']:
            print('failed realizations    : %d / %d' % (mc['nfail'], args.mc))
        print('[OIII] 5007/4959 ratio : %.4f +/- %.4f' % mc['oiii_ratio'])
        print('[OIII]/Hb narrow ratio : %.4f +/- %.4f' % mc['ratio'])
        print('narrow Hb center       : %.2f +/- %.2f'
              % (5007.0 - mc['shift'][0], mc['shift'][1]))
        sig_total = mc['sig_narrow_total']
    else:
        sig_total = np.zeros_like(wave)

    err_out = np.sqrt(err**2 + sig_total**2)

    # narrow-subtracted spectrum, same 3-column format as the original
    outname = spec + '.subnarrow'
    with open(outname, 'w') as f:
        for i in range(wave.size):
            f.write('%f  %e  %e\n'
                    % (wave[i], r['flux_sub'][i] * UNIT, err_out[i] * UNIT))
    print('subtracted spectrum    : %s' % outname)

    # narrow model and components on the full grid
    zeros = np.zeros_like(wave)
    sig = {k: (mc['sig_' + k] if mc is not None else zeros)
           for k in ('narrow_total', 'narrow_4959', 'narrow_5007',
                     'narrow_hb')}
    profname = '%s_narrow_profile.txt' % obj_name
    header = ('wave  narrow_total  sig_total  oiii5007  sig_5007  '
              'oiii4959  sig_4959  narrow_hb  sig_hb   (flux x %g)' % UNIT)
    np.savetxt(profname,
               np.column_stack([wave,
                                r['narrow_total'] * UNIT,
                                sig['narrow_total'] * UNIT,
                                r['narrow_5007'] * UNIT,
                                sig['narrow_5007'] * UNIT,
                                r['narrow_4959'] * UNIT,
                                sig['narrow_4959'] * UNIT,
                                r['narrow_hb'] * UNIT,
                                sig['narrow_hb'] * UNIT]),
               fmt='%f  ' + '  '.join(['%e'] * 8), header=header)
    print('narrow profile         : %s' % profname)

    # clean [OIII]5007 template
    sig_t = mc['sig_flux_t'] if mc is not None else np.zeros_like(r['wave_t'])
    tmplname = '%s_oiii5007_template.txt' % obj_name
    np.savetxt(tmplname,
               np.column_stack([r['wave_t'], r['flux_t'] * UNIT,
                                r['err_t'] * UNIT, sig_t * UNIT]),
               fmt='%f  %e  %e  %e',
               header='wave  flux  err_data  sig_mc   (flux x %g)' % UNIT)
    print('[OIII]5007 template    : %s' % tmplname)

    if not args.no_plot:
        make_plot(wave, flux, err, p, r, mc, obj_name,
                  '%s_subtract_narrow_mean.pdf' % obj_name, args.show)


if __name__ == '__main__':
    main()
