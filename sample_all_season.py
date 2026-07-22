#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Run PYCCF on all season light-curve pairs, using all CPU cores.

For every (continuum, line) pair listed in JOBS this script:
  1. computes the CCF of the real data with PYCCF.peakcent,
  2. splits the NSIM FR/RSS Monte Carlo iterations into chunks and runs them
     in parallel on a multiprocessing pool (every chunk gets its own random
     seed, so the concatenated CCCD/CCPD is statistically identical to a
     single serial xcor_mc call),
  3. writes <label>_centtab.dat, <label>_peaktab.dat, <label>_ccf.dat and
     <label>_results_plot.pdf, plus a summary table all_season_lags.txt.

Run it as a normal single-process script (NOT with mpirun):
    python sample_all_season.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI needed; safe for headless/parallel use
from matplotlib import pyplot as plt, gridspec
from multiprocessing import Pool, cpu_count
from scipy import stats

import PYCCF as myccf

########################################
# Light-curve pairs: (continuum file, line file, output label[, lag_range])
# lag_range is optional: give a pair its own [tlagmin, tlagmax] lag prior,
# otherwise the default LAG_RANGE below is used, e.g.
#     ("NGC3227_continuum_low.dat", "NGC3227_Hbeta_low.dat", "hbeta_low", [-5, 25]),
# Edit this list to add/remove seasons or lines.
########################################
JOBS = [
    # s1
    ("NGC3227_continuum.dat",        "NGC3227_Hbeta.dat",        "hbeta_all",    [-5, 20]),
    # s2
    ("NGC3227_continuum_high.dat",   "NGC3227_Hbeta_high.dat",   "hbeta_high",   [-5, 15]),
    # s3
    ("NGC3227_continuum_low.dat",    "NGC3227_Hbeta_low.dat",    "hbeta_low",    [-5, 25]),
    # s4
    ("NGC3227_continuum_high.dat",   "NGC3227_he_high.dat",      "he_high",      [-5, 10]),
]

########################################
# CCF settings (same meaning as in sample_runcode.py)
########################################
LAG_RANGE = [-5, 15]   # default lag search range (days) for jobs without their own
INTERP = 1             # interpolation step (days)
NSIM = 20000           # total Monte Carlo iterations per pair
MCMODE = 0             # 0 = FR+RSS, 1 = RSS only, 2 = FR only
CHUNK_SIZE = 1000      # MC iterations per parallel task
NCORES = cpu_count()   # number of worker processes
OUTPUT_DIR = './results/'

PERCLIM = 84.1344746   # 1-sigma percentile


def run_mc_chunk(args):
    """One Monte Carlo chunk; executed in a worker process."""
    seed, jd1, f1, e1, jd2, f2, e2, nsim_chunk, sigmode, lag_range = args
    # xcor_mc draws from the global numpy RNG, so each chunk must be
    # seeded explicitly (with fork-started workers all children would
    # otherwise inherit the same state and produce identical chunks).
    np.random.seed(seed)
    res = myccf.xcor_mc(jd1, f1, np.abs(e1), jd2, f2, np.abs(e2),
                        lag_range[0], lag_range[1], INTERP,
                        nsim=nsim_chunk, mcmode=MCMODE, sigmode=sigmode)
    tlags_peak, tlags_centroid = res[0], res[1]
    nfail_peak, nfail_centroid = res[3], res[5]
    return tlags_peak, tlags_centroid, nfail_peak, nfail_centroid


def make_plot(pair, tlags_centroid, centau):
    """Light curves + CCF + CCCD figure, same layout as sample_runcode.py."""
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(pair['jd_con'] - 2458800, pair['con'], yerr=pair['econ'],
                 marker='.', linestyle='none', c='k', markersize=9)
    ax1.set_ylabel('$F_{5100\\AA}$')

    ax1_2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax1_2.errorbar(pair['jd_line'] - 2458800, pair['line'], yerr=pair['eline'],
                   marker='.', linestyle='none', c='k', markersize=9)
    ax1_2.set_ylabel('$F_{line}$')
    ax1_2.set_xlabel('JD-2458800')

    xmin, xmax = pair['lag_range'][0], pair['lag_range'][1]

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(pair['ccf_lag'], pair['ccf_r'])
    ax2.axvline(0, color='grey', linestyle='--')
    ax2.axvline(centau, color='orange', linestyle='--')
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('CCF')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax3.hist(tlags_centroid, bins=10, color='orange', density=True)
    ax3.axvline(0, color='grey', linestyle='--')
    ax3.axvline(centau, color='orange', linestyle='--')
    ax3.set_ylabel('CCCD')
    ax3.set_xlim(xmin, xmax)
    ax3.set_xlabel('Lag (days)')
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + pair['label'] + '_results_plot.pdf',
                format='pdf', orientation='landscape', bbox_inches='tight')
    plt.close(fig)


def main():
    ########################################
    # Load light curves and run the real-data CCF for each pair
    ########################################
    pairs = []
    for job in JOBS:
        con_file, line_file, label = job[0], job[1], job[2]
        lag_range = job[3] if len(job) > 3 else LAG_RANGE

        data = np.loadtxt(con_file)
        jd_con, con, econ = data[:, 0], data[:, 1], data[:, 2]
        data = np.loadtxt(line_file)
        jd_line, line, eline = data[:, 0], data[:, 1], data[:, 2]

        (tlag_peak, status_peak, tlag_centroid, status_centroid,
         ccf_pack, max_rval, status_rval, pval) = myccf.peakcent(
            jd_con, con, jd_line, line, lag_range[0], lag_range[1], INTERP)

        pairs.append(dict(
            label=label,
            lag_range=lag_range,
            jd_con=jd_con, con=con, econ=econ,
            jd_line=jd_line, line=line, eline=eline,
            ccf_r=ccf_pack[0], ccf_lag=ccf_pack[1],
            rmax=ccf_pack[0].max(),
            sigmode=0.8 * ccf_pack[0].max(),  # same threshold as sample_runcode.py
        ))

    ########################################
    # Build one task list for ALL pairs and run it on the pool
    ########################################
    tasks = []
    task_pair_idx = []
    seed_seq = np.random.SeedSequence()
    for ip, p in enumerate(pairs):
        nsim_left = NSIM
        while nsim_left > 0:
            n = min(CHUNK_SIZE, nsim_left)
            nsim_left -= n
            seed = int(seed_seq.spawn(1)[0].generate_state(1)[0] % (2**31 - 1))
            tasks.append((seed, p['jd_con'], p['con'], p['econ'],
                          p['jd_line'], p['line'], p['eline'],
                          n, p['sigmode'], p['lag_range']))
            task_pair_idx.append(ip)

    print('Running %d pairs x %d simulations as %d tasks on %d cores...'
          % (len(pairs), NSIM, len(tasks), NCORES))
    with Pool(NCORES) as pool:
        results = pool.map(run_mc_chunk, tasks)

    ########################################
    # Regroup chunks per pair, write outputs, plot
    ########################################
    summary = open(OUTPUT_DIR + 'all_season_lags.txt', 'w')
    summary.write('# label  cent  -err  +err  peak  -err  +err  rmax  nfail_cent  nfail_peak\n')

    for ip, p in enumerate(pairs):
        chunks = [results[i] for i in range(len(results)) if task_pair_idx[i] == ip]
        tlags_peak = np.concatenate([c[0] for c in chunks])
        tlags_centroid = np.concatenate([c[1] for c in chunks])
        nfail_peak = sum(c[2] for c in chunks)
        nfail_centroid = sum(c[3] for c in chunks)

        centau = stats.scoreatpercentile(tlags_centroid, 50)
        centau_uperr = stats.scoreatpercentile(tlags_centroid, PERCLIM) - centau
        centau_loerr = centau - stats.scoreatpercentile(tlags_centroid, 100. - PERCLIM)

        peaktau = stats.scoreatpercentile(tlags_peak, 50)
        peaktau_uperr = stats.scoreatpercentile(tlags_peak, PERCLIM) - peaktau
        peaktau_loerr = peaktau - stats.scoreatpercentile(tlags_peak, 100. - PERCLIM)

        print('%-14s Centroid: %8.3f (+%6.3f -%6.3f)   Peak: %8.3f (+%6.3f -%6.3f)   '
              'r_max=%.3f   failed cent/peak: %d/%d'
              % (p['label'], centau, centau_uperr, centau_loerr,
                 peaktau, peaktau_uperr, peaktau_loerr,
                 p['rmax'], nfail_centroid, nfail_peak))
        summary.write('%s  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %d  %d\n'
                      % (p['label'], centau, centau_loerr, centau_uperr,
                         peaktau, peaktau_loerr, peaktau_uperr,
                         p['rmax'], nfail_centroid, nfail_peak))

        np.savetxt(OUTPUT_DIR + p['label'] + '_centtab.dat', tlags_centroid, fmt='%5.5f')
        np.savetxt(OUTPUT_DIR + p['label'] + '_peaktab.dat', tlags_peak, fmt='%5.5f')
        np.savetxt(OUTPUT_DIR + p['label'] + '_ccf.dat',
                   np.column_stack([p['ccf_lag'], p['ccf_r']]), fmt='%5.5f')

        make_plot(p, tlags_centroid, centau)

    summary.close()
    print('Done. Summary written to all_season_lags.txt')


if __name__ == '__main__':
    main()
