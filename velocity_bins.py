#!/usr/bin/env python3
"""
Velocity bins of the broad Hbeta profile for velocity-resolved time lags
(CLI version).

Same pipeline as velocity_bins.ipynb -- keep the two in sync.
Input: rms spectrum (spec_rms_<line>.txt) or narrow-subtracted mean
spectrum (spec_mean_<line>.txt.subnarrow), REST-FRAME wavelength,
2 or 3 columns: wave flux [err].

Method (Denney et al. 2009, ApJ 704, L80; MAHA I Sec. 4.4 + App. B)
  P(lambda) = flux above a linear continuum interpolated between two
  line-free windows (same convention as measure_linewidth).
  Zero velocity = flux centroid of the narrow-Hbeta model column of
  <obj>_narrow_profile.txt (written by subtract_narrow_mean), so
  v = c (lambda - lam0) / lam0.

  Two binning schemes, both computed every run, over the same
  [--line LO HI] window (= the Hbeta integration window):
    eqflux : N bins of equal integrated P(lambda). Edges are the first
             crossings of the cumulative flux through k/N of the net
             window flux; raw (possibly negative) pixels are kept so
             the total matches the light-curve integration downstream.
             Edges whose level is crossed more than once (noise dips
             below zero) are flagged in the output.
    eqwidth: N bins of equal velocity width (uniform edges).

  Number of bins (rule of thumb agreed 2026-07-19): default
    N = floor(line-window velocity width / LSF)
  with LSF the instrumental broadening FWHM (MAHA: 925 km/s), and a
  WARNING whenever any resulting bin is narrower than the LSF --
  such bins are not kinematically independent of their neighbours.

Outputs (obj tag = --obj, default: current directory name; spec tag =
--tag, default guessed from the input file name: rms / mean / spec)
  <obj>_velbins_<tag>_eqflux.txt    edges + per-bin flux table
  <obj>_velbins_<tag>_eqwidth.txt   same for the equal-width scheme
  <obj>_velbins_<tag>.pdf           profile + edges, both schemes

Usage
-----
  python velocity_bins.py spec_rms_hbeta.txt \\
      --line 4800 4950 --contil 4780 4800 --contir 4950 4970 \\
      --narrow-profile NGC3227_narrow_profile.txt \\
      [--nbins N] [--lsf 925] [--v0-wave 4861.5] \\
      [--obj NAME] [--tag rms] [--no-plot] [--show]
"""

import argparse
import os

import numpy as np

UNIT = 1.0e-14
C_KMS = 299792.458

# Okabe-Ito colorblind-safe palette, fixed assignment
C_DATA = '#000000'       # observed spectrum
C_MODEL = '#E69F00'      # bin edges
C_BROAD = '#0072B2'      # equal-flux scheme
C_NARROW = '#009E73'     # zero-velocity line
C_SUB = '#D55E00'        # line profile P
C_CON = '#CC79A7'        # continuum / background lines


def read_spec(filename):
    data = np.loadtxt(filename)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError('need at least 2 columns: wave flux [err]')
    err = data[:, 2] if data.shape[1] > 2 else np.zeros(data.shape[0])
    return data[:, 0], data[:, 1], err


def window_median(wave, flux, lim1, lim2):
    """Median flux and mean wavelength inside [lim1, lim2]."""
    idx = np.where((wave >= lim1) & (wave <= lim2))[0]
    if idx.size == 0:
        raise ValueError('empty window %.1f-%.1f' % (lim1, lim2))
    return np.median(flux[idx]), np.mean(wave[idx])


def linear_through(w1, f1, w2, f2, wave):
    return f1 + (f2 - f1) / (w2 - w1) * (wave - w1)


def continuum_model(wave, flux, p):
    """Linear continuum through the two window medians; zero if no windows."""
    if p.get('conti_left') is None or p.get('conti_right') is None:
        return np.zeros_like(wave)
    fl, wl = window_median(wave, flux, *p['conti_left'])
    fr, wr = window_median(wave, flux, *p['conti_right'])
    return linear_through(wl, fl, wr, fr, wave)


def narrow_hb_centroid(profile_file):
    """Flux centroid of the narrow-Hbeta model in <obj>_narrow_profile.txt.

    Column 7 (0-based) is narrow_hb, per the header written by
    subtract_narrow_mean. The centroid of this model defines v = 0.
    """
    d = np.loadtxt(profile_file)
    if d.ndim != 2 or d.shape[1] < 8:
        raise ValueError('%s: expected the 9-column narrow_profile format'
                         % profile_file)
    w, f = d[:, 0], d[:, 7]
    norm = np.trapz(f, w)
    if norm <= 0.0:
        raise ValueError('%s: narrow_hb column has no positive flux -- '
                         'give --v0-wave instead' % profile_file)
    return np.trapz(w * f, w) / norm


def wave_to_vel(wave, lam0):
    return C_KMS * (np.asarray(wave, float) - lam0) / lam0


def default_nbins(line_win, lam0, lsf_kms):
    """Rule of thumb: floor(line-window velocity width / LSF), >= 2."""
    dv = C_KMS * (line_win[1] - line_win[0]) / lam0
    return max(int(dv // lsf_kms), 2)


def profile_in_window(wave, flux, p):
    """Continuum-subtract and slice the line window."""
    fcon = continuum_model(wave, flux, p)
    prof = flux - fcon
    lo, hi = p['line_win']
    idx = np.where((wave >= lo) & (wave <= hi))[0]
    if idx.size < 5:
        raise ValueError('line window has %d points' % idx.size)
    return wave[idx], prof[idx], fcon


def cumulative_flux(w, pr):
    """Trapezoidal cumulative integral of pr over w; C[0] = 0."""
    dc = 0.5 * (pr[1:] + pr[:-1]) * np.diff(w)
    return np.concatenate([[0.0], np.cumsum(dc)])


def equal_flux_edges(w, pr, nbins):
    """Edges of nbins bins with equal integrated flux inside the window.

    Interior edge k is the FIRST crossing of the cumulative flux C
    through the level k/N of the net window flux (raw fluxes kept:
    negative noise pixels contribute their flux but are never edges).
    Returns (edges array of length nbins+1, list of multi-crossing
    flags for the nbins-1 interior edges).
    """
    c = cumulative_flux(w, pr)
    total = c[-1]
    if total <= 0.0:
        raise ValueError('non-positive net flux in the line window')
    edges = [w[0]]
    flagged = []
    for k in range(1, nbins):
        level = total * k / nbins
        above = c >= level
        i = int(np.argmax(above))       # first index with C >= level
        edges.append(w[i - 1] + (level - c[i - 1]) * (w[i] - w[i - 1])
                     / (c[i] - c[i - 1]))
        ncross = int(np.count_nonzero(np.diff(above.astype(int)) == 1))
        flagged.append(ncross > 1)
    edges.append(w[-1])
    return np.asarray(edges), flagged


def equal_width_edges(w_lo, w_hi, nbins):
    """Edges of nbins bins of equal velocity width.

    v is linear in lambda, so uniform in wavelength == uniform in
    velocity.
    """
    return np.linspace(w_lo, w_hi, nbins + 1)


def bin_stats(w, pr, edges, lam0):
    """Per-bin integrated flux, flux fraction and flux-weighted velocity.

    Each bin is integrated on the pixel grid plus linearly interpolated
    edge points, so the per-bin fluxes partition the window total
    exactly (trapezoids split exactly at interpolated points).
    """
    total = cumulative_flux(w, pr)[-1]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (w > lo) & (w < hi)
        wseg = np.concatenate([[lo], w[inside], [hi]])
        pseg = np.interp(wseg, w, pr)
        fl = np.trapz(pseg, wseg)
        vseg = wave_to_vel(wseg, lam0)
        if fl != 0.0:
            vcen = np.trapz(vseg * pseg, wseg) / fl
        else:
            vcen = 0.5 * (vseg[0] + vseg[-1])
        rows.append({'flux': fl, 'frac': fl / total, 'vcen': vcen})
    return rows


def narrow_bin_indices(edges, lam0, lsf_kms):
    """Indices of bins narrower than the LSF (not independent bins)."""
    widths = np.diff(wave_to_vel(edges, lam0))
    return widths, [i for i, wd in enumerate(widths) if wd < lsf_kms]


def compute_bins(wave, flux, p, lam0, nbins, lsf_kms):
    """Both schemes on one spectrum. Returns dict of per-scheme results."""
    w, pr, fcon = profile_in_window(wave, flux, p)
    out = {'w': w, 'pr': pr, 'fcon': fcon, 'lam0': lam0, 'nbins': nbins}
    eq_edges, eq_flags = equal_flux_edges(w, pr, nbins)
    ew_edges = equal_width_edges(w[0], w[-1], nbins)
    for name, edges, flags in (('eqflux', eq_edges, eq_flags),
                               ('eqwidth', ew_edges, [False] * (nbins - 1))):
        widths, narrow = narrow_bin_indices(edges, lam0, lsf_kms)
        out[name] = {'edges': edges, 'v_edges': wave_to_vel(edges, lam0),
                     'widths': widths, 'narrow': narrow,
                     'multicross': flags,
                     'bins': bin_stats(w, pr, edges, lam0)}
    return out


def write_bins(res, scheme, spec, lsf_kms, outname):
    """Edge table: one row per bin, header with the run metadata."""
    r = res[scheme]
    # a bin is marked if either of its edges came from a multi-crossing
    flags = []
    for i in range(res['nbins']):
        mc = (i > 0 and r['multicross'][i - 1]) or \
             (i < res['nbins'] - 1 and r['multicross'][i])
        tag = []
        if mc:
            tag.append('multicross')
        if i in r['narrow']:
            tag.append('lt_lsf')
        flags.append(','.join(tag) if tag else '-')
    header = ('spec %s | scheme %s | nbins %d | lam0 %.4f A (narrow Hbeta '
              'centroid, v=0) | lsf %.0f km/s | window %.1f-%.1f A\n'
              'bin  v_lo_kms  v_hi_kms  width_kms  lam_lo_A  lam_hi_A  '
              'vcen_kms  flux  frac  flag'
              % (spec, scheme, res['nbins'], res['lam0'], lsf_kms,
                 r['edges'][0], r['edges'][-1]))
    lines = []
    for i, b in enumerate(r['bins']):
        lines.append('%3d  %9.1f  %9.1f  %9.1f  %9.3f  %9.3f  %9.1f  '
                     '%12.5e  %7.4f  %s'
                     % (i + 1, r['v_edges'][i], r['v_edges'][i + 1],
                        r['widths'][i], r['edges'][i], r['edges'][i + 1],
                        b['vcen'], b['flux'] * UNIT, b['frac'], flags[i]))
    with open(outname, 'w') as f:
        for h in header.split('\n'):
            f.write('# %s\n' % h)
        f.write('\n'.join(lines) + '\n')
    print('%s edges              : %s' % (scheme.ljust(7), outname))


def make_plot(res, lsf_kms, obj_name, tag, outfile, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    v = wave_to_vel(res['w'], res['lam0'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    fig.suptitle('%s  (%s spectrum)  N=%d  v=0 at %.2f $\\AA$'
                 % (obj_name, tag, res['nbins'], res['lam0']))
    for ax, scheme, label in ((axes[0], 'eqflux', 'equal flux per bin'),
                              (axes[1], 'eqwidth', 'equal velocity width')):
        r = res[scheme]
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.plot(v, res['pr'], color=C_SUB, lw=1.2, label='P($\\lambda$)')
        ax.axhline(0.0, color='gray', lw=0.5)
        ax.axvline(0.0, color=C_NARROW, lw=1.0, ls='--', label='v = 0')
        for i, ve in enumerate(r['v_edges']):
            ax.axvline(ve, color=C_MODEL, lw=0.9,
                       ls=':' if 0 < i < res['nbins'] else '-')
        top = np.max(res['pr'])
        for i, b in enumerate(r['bins']):
            vc = 0.5 * (r['v_edges'][i] + r['v_edges'][i + 1])
            note = '%.0f%%' % (100.0 * b['frac'])
            if i in r['narrow']:
                note += '\n<LSF'
            ax.text(vc, 1.02 * top, note, ha='center', va='bottom',
                    fontsize=7, color=C_BROAD)
        # LSF scale bar
        ax.plot([r['v_edges'][0], r['v_edges'][0] + lsf_kms],
                [0.92 * top] * 2, color=C_DATA, lw=2.5)
        ax.text(r['v_edges'][0] + 0.5 * lsf_kms, 0.94 * top,
                'LSF %.0f km/s' % lsf_kms, ha='center', va='bottom',
                fontsize=7)
        ax.set_ylim(top=1.14 * top)
        ax.set_ylabel('flux (%.0e erg/s/cm$^2$/$\\AA$)' % UNIT)
        ax.set_title(label, fontsize=10)
    axes[1].set_xlabel('velocity (km s$^{-1}$)')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(outfile, format='pdf', bbox_inches='tight')
    print('diagnostic plot        : %s' % outfile)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Velocity bins of the broad Hbeta profile for '
                    'velocity-resolved lags: equal-flux and equal-width '
                    'schemes (Denney+09 / MAHA), v=0 at the narrow-Hbeta '
                    'centroid.')
    ap.add_argument('spec', help='rms or narrow-subtracted mean spectrum: '
                                 'wave flux [err], rest-frame')
    ap.add_argument('--line', nargs=2, type=float, required=True,
                    metavar=('LO', 'HI'),
                    help='line window = Hbeta integration window (outer '
                         'bin boundaries)')
    ap.add_argument('--contil', nargs=2, type=float, default=None,
                    metavar=('LO', 'HI'),
                    help='left continuum window (omit both: continuum = 0)')
    ap.add_argument('--contir', nargs=2, type=float, default=None,
                    metavar=('LO', 'HI'), help='right continuum window')
    ap.add_argument('--narrow-profile', default=None,
                    help='<obj>_narrow_profile.txt from subtract_narrow_mean;'
                         ' v=0 = centroid of its narrow_hb column')
    ap.add_argument('--v0-wave', type=float, default=None,
                    help='zero-velocity wavelength in A (alternative to '
                         '--narrow-profile)')
    ap.add_argument('--nbins', type=int, default=None,
                    help='number of bins (default: floor(window width / '
                         'LSF))')
    ap.add_argument('--lsf', type=float, default=925.0,
                    help='instrumental LSF FWHM in km/s; sets the default '
                         'nbins and the minimum-width warning (MAHA: 925)')
    ap.add_argument('--obj', default=None,
                    help='output name tag (default: current directory name)')
    ap.add_argument('--tag', default=None,
                    help='spectrum tag in output names (default: rms/mean/'
                         'spec guessed from the file name)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--show', action='store_true',
                    help='open the plot window (default: save PDF only)')
    args = ap.parse_args()

    if (args.contil is None) != (args.contir is None):
        ap.error('--contil and --contir must be given together')
    if (args.narrow_profile is None) == (args.v0_wave is None):
        ap.error('give exactly one of --narrow-profile / --v0-wave')
    if args.nbins is None and args.lsf <= 0.0:
        ap.error('--nbins is required when --lsf is 0')

    obj_name = args.obj or os.path.basename(os.getcwd())
    base = os.path.basename(args.spec).lower()
    tag = args.tag or ('rms' if 'rms' in base
                       else 'mean' if 'mean' in base else 'spec')
    p = {'line_win': tuple(args.line),
         'conti_left': tuple(args.contil) if args.contil else None,
         'conti_right': tuple(args.contir) if args.contir else None}

    wave, flux, err = read_spec(args.spec)
    flux = flux / UNIT

    if p['conti_left'] is None:
        print('no continuum windows given -> continuum assumed 0')

    if args.narrow_profile is not None:
        lam0 = narrow_hb_centroid(args.narrow_profile)
        print('v = 0 at %.4f A (narrow-Hbeta centroid of %s)'
              % (lam0, args.narrow_profile))
    else:
        lam0 = args.v0_wave
        print('v = 0 at %.4f A (--v0-wave)' % lam0)

    nbins = args.nbins
    if nbins is None:
        nbins = default_nbins(p['line_win'], lam0, args.lsf)
        print('nbins = %d  (default: floor(window %.0f km/s / LSF %.0f '
              'km/s))' % (nbins, C_KMS * (p['line_win'][1]
                          - p['line_win'][0]) / lam0, args.lsf))
    else:
        print('nbins = %d  (user)' % nbins)

    res = compute_bins(wave, flux, p, lam0, nbins, args.lsf)

    for scheme in ('eqflux', 'eqwidth'):
        r = res[scheme]
        print('--- %s ---' % scheme)
        for i, b in enumerate(r['bins']):
            print('bin %2d  v %8.1f .. %8.1f  width %7.1f km/s  '
                  'frac %.4f' % (i + 1, r['v_edges'][i],
                                 r['v_edges'][i + 1], r['widths'][i],
                                 b['frac']))
        if r['narrow']:
            print('WARNING: bin(s) %s narrower than the LSF (%.0f km/s) -- '
                  'not kinematically independent; consider fewer bins'
                  % (', '.join(str(i + 1) for i in r['narrow']), args.lsf))
        if any(r['multicross']):
            ks = [str(k + 1) for k, m in enumerate(r['multicross']) if m]
            print('NOTE: cumulative flux crossed level(s) %s more than once '
                  '(noise dips below zero); first crossing used'
                  % ', '.join(ks))
        write_bins(res, scheme, args.spec, args.lsf,
                   '%s_velbins_%s_%s.txt' % (obj_name, tag, scheme))

    if not args.no_plot:
        make_plot(res, args.lsf, obj_name, tag,
                  '%s_velbins_%s.pdf' % (obj_name, tag), args.show)


if __name__ == '__main__':
    main()
