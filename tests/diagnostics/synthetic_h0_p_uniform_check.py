"""Reviewer G2: gene-level p uniform on synthetic H0 data.

Builds 5000 ions × 4 mice × 3 microdiss × 2 arms with NO biology, runs
the full paired ion test on every ion, and checks that the analytical
p's are uniform on [0, 1].
"""
import sys

sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant')
sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant/tests/unit_tests')
import numpy as np
from scipy.stats import kstest
import alphaquant.diffquant.background_distributions_paired as aqbg_paired
import alphaquant.diffquant.diff_analysis as aqdiff
import alphaquant.diffquant.condpair_analysis as aq_condpair
import test_paired_phase2 as t


def main():
    df1, df2, triples = t._make_bm_mk_shaped_frames(
        n_mice=4, n_microdiss=3, n_ions=5000,
        sigma_within_cell=0.18, sigma_between_subj=0.4,
        fc_signal=0.0, spike_ions=0, seed=2026,
    )
    pc = aqbg_paired.PairedConditionBackgrounds(df1, df2, triples, p2z={})

    p_vals = []
    for ion in df1.index:
        bg = pc.get_paired_diff_dist(ion)
        if bg is None:
            continue
        pairs = aq_condpair._build_paired_subject_pairs_for_ion(
            ion, triples, df1, df2)
        di = aqdiff.DifferentialIonPaired(
            pairs, bg, name=ion, outlier_correction=False)
        if di.usable:
            p_vals.append(di.p_val)
    p_vals = np.array(p_vals)
    print(f'gene/ion-level p stats under H0 (n={len(p_vals)}):')
    print(f'  mean p   = {np.mean(p_vals):.4f}  (expected 0.5)')
    print(f'  median p = {np.median(p_vals):.4f}  (expected 0.5)')
    print(f'  fraction p<0.05  = {np.mean(p_vals < 0.05):.4f}  (expected 0.05)')
    print(f'  fraction p<0.01  = {np.mean(p_vals < 0.01):.4f}  (expected 0.01)')
    print(f'  fraction p<0.001 = {np.mean(p_vals < 0.001):.4f}  (expected 0.001)')
    ks = kstest(p_vals, 'uniform')
    print(f'KS test vs uniform[0,1]: stat={ks.statistic:.4f}, p={ks.pvalue:.4g}')


if __name__ == '__main__':
    main()
