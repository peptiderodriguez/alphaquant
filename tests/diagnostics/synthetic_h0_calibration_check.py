"""Synthetic H0 calibration check for the C1-fixed paired empirical null.

Builds a 5000-ion synthetic dataset with NO biology (fc_signal=0), runs the
paired backgrounds, and checks that the per-(subject, ion) z's returned by
calc_zscore_from_fc follow ~ N(0, 1). This is the Type-I calibration the
reviewer's G2 finding asks for, on a clean H0.

If post-lookup z mean ~ 0 and SD ~ 1 with a tight tail rate, the C1 fix's
calibration claim holds. (Real bm_mk data CONTAINS biology, so its
empirical null is leptokurtic — the QQ plot vs N(0, SD_bin) misleads
observers into seeing a "bug" that is actually correct behavior under
real biological signal contamination.)
"""
from __future__ import annotations
import sys

import numpy as np

sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant')
import alphaquant.diffquant.background_distributions_paired as aqbg_paired
sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/alphaquant/tests/unit_tests')
import test_paired_phase2 as t


def main() -> None:
    df1, df2, triples = t._make_bm_mk_shaped_frames(
        n_mice=4, n_microdiss=3, n_ions=5000,
        sigma_within_cell=0.18, sigma_between_subj=0.4,
        fc_signal=0.0, spike_ions=0, seed=2026,
    )
    pc = aqbg_paired.PairedConditionBackgrounds(df1, df2, triples, p2z={})
    print(f'Built {len(pc.backgrounds)} bins')
    for i, bg in enumerate(pc.backgrounds):
        print(f'  bin {i}: SD={bg.SD:.4f}, ions={len(bg.ions)}')

    zs = []
    for ion in df1.index:
        bg = pc.get_paired_diff_dist(ion)
        if bg is None or bg.SD == 0:
            continue
        for subj, c1, c2 in triples:
            m1 = float(np.nanmean(df1.loc[ion, c1].to_numpy(dtype=float)))
            m2 = float(np.nanmean(df2.loc[ion, c2].to_numpy(dtype=float)))
            z = bg.calc_zscore_from_fc(m1 - m2)
            zs.append(z)
    zs = np.array(zs)
    print(f'\nPost-lookup z stats (n={len(zs)}):')
    print(f'  mean = {np.mean(zs):.4f}  (expected 0.000)')
    print(f'  SD   = {np.std(zs):.4f}  (expected 1.000)')
    print(f'  |z|>1.96 fraction = {np.mean(np.abs(zs) > 1.96):.4f}  (expected 0.0500)')
    print(f'  |z|>2.58 fraction = {np.mean(np.abs(zs) > 2.58):.4f}  (expected 0.0100)')


if __name__ == '__main__':
    main()
