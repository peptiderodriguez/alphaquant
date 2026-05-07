import alphaquant.diffquant.background_distributions as aqbg
import alphaquant.diffquant.background_distributions_paired as aqbg_paired
import alphaquant.diffquant.diff_analysis as aqdiff
import alphaquant.diffquant.paired_permutation_null as aqperm
import alphaquant.norm.normalization as aqnorm
import alphaquant.plotting.pairwise as aq_plot_pairwise
import alphaquant.diffquant.diffutils as aqutils
import alphaquant.cluster.cluster_ions as aqclust
import alphaquant.cluster.correlation_aware_stouffer as aq_corr
import alphaquant.classify.classify_precursors as aq_class_precursors
import alphaquant.cluster.ml_reorder as aq_clust_mlreorder
import alphaquant.tables.diffquant_table as aq_tablewriter_protein
import alphaquant.tables.proteoformtable as aq_tablewriter_proteoform
import alphaquant.tables.misctables as aq_tablewriter_runconfig
import alphaquant.cluster.cluster_utils as aqclust_utils
import alphaquant.cluster.cluster_missingval as aq_clust_missingval
import alphaquant.cluster.outlier_filtering as aq_clust_outlier

import pandas as pd
import numpy as np
import os

from collections import defaultdict

import alphaquant.config.config as aqconfig
import logging
aqconfig.setup_logging()
LOGGER = logging.getLogger(__name__)

def analyze_condpair(*,runconfig, condpair):
    """Main workflow orchestration for differential analysis of a condition pair.

    This function coordinates the complete analysis pipeline for comparing two conditions:
    1. Loads and filters data for the two conditions
    2. Performs normalization (within and between conditions)
    3. Creates empirical background distributions
    4. Computes ion-level differential statistics
    5. Builds hierarchical trees and performs clustering to identify proteoforms
    6. Applies machine learning quality scoring (if enabled)
    7. Filters outlier peptides (if enabled)
    8. Generates output tables with FDR correction
    9. Creates visualization plots

    Args:
        runconfig: Configuration object containing all analysis parameters (see run_pipeline docstring)
        condpair: Tuple of (condition1_name, condition2_name) to compare

    Returns:
        tuple: (results_df, peptide_df) - DataFrames with protein and peptide-level results
    """
    LOGGER.info(f"start processeing condpair {condpair}")
    prot2diffions = defaultdict(list) #per default maps any key to empty list
    prot2missingval_diffions = defaultdict(list)
    p2z = {}
    ion2clust = {}
    protnodes = []
    protnodes_missingval = []

    input_df_local = get_unnormed_df_condpair(input_file=runconfig.input_file_reformat, samplemap_df=runconfig.samplemap_df, condpair=condpair, file_has_alphaquant_format = runconfig.file_has_alphaquant_format)
    pep2prot = dict(zip(input_df_local.index, input_df_local['protein']))
    c1_samples, c2_samples = aqutils.get_samples_used_from_samplemap_df(runconfig.samplemap_df, condpair[0], condpair[1])

    try:
        df_c1, df_c2 = get_per_condition_dataframes(c1_samples, c2_samples, input_df_local, min_valid_values=runconfig.min_valid_values, valid_values_filter_mode=runconfig.valid_values_filter_mode, min_valid_values_c1=runconfig.min_valid_values_c1, min_valid_values_c2=runconfig.min_valid_values_c2)
    except Exception as e:
        LOGGER.info(e)
        return

    df_c1_normed, df_c2_normed = aqnorm.normalize_if_specified(df_c1 = df_c1, df_c2 = df_c2, c1_samples = c1_samples, c2_samples = c2_samples, normalize_within_conds = runconfig.normalize, normalize_between_conds = runconfig.normalize,
    runtime_plots = runconfig.runtime_plots, protein_subset_for_normalization_file=runconfig.protein_subset_for_normalization_file, pep2prot = pep2prot)#, "./test_data/normed_intensities.tsv")

    if runconfig.results_dir != None:
        write_out_normed_df(df_c1_normed, df_c2_normed, pep2prot, runconfig.results_dir, condpair)
    normed_c1 = aqbg.ConditionBackgrounds(df_c1_normed, p2z)
    normed_c2 = aqbg.ConditionBackgrounds(df_c2_normed, p2z)

    ions_to_check = normed_c1.ion2nonNanvals.keys() & normed_c2.ion2nonNanvals.keys()
    ions_to_check = sorted(ions_to_check)

    # ── paired-design dispatch (Phase 1 plumbing) ────────────────────────
    # When ion_test_method='paired_diffdist' is requested, check if THIS
    # contrast is paired-eligible (subjects appear in both arms). Eligible
    # contrasts route to DifferentialIonPaired; ineligible contrasts (e.g.,
    # mixing a paired and an unpaired contrast in the same run) WARN and
    # gracefully fall back to the unpaired DifferentialIon path. This keeps
    # mixed paired/unpaired runs working — only paired-eligible contrasts
    # actually use the paired test.

    # Validate ion_test_method to catch typos (e.g., 'paired_difdist').
    # Without this, a typo would silently fall through to the unpaired
    # path, hiding the user's bug. Allowed methods are extended via
    # this list (no enum to avoid touching the runconfig dataclass shape).
    _allowed_ion_test_methods = {'diffdist', 'ttest', 'paired_diffdist'}
    if runconfig.ion_test_method not in _allowed_ion_test_methods:
        raise ValueError(
            f"runconfig.ion_test_method={runconfig.ion_test_method!r} is not "
            f"recognized. Allowed values: {sorted(_allowed_ion_test_methods)}. "
            f"Common typo: 'paired_difdist' should be 'paired_diffdist'.")

    use_paired = False
    paired_triples = None
    paired_normed = None
    if runconfig.ion_test_method == 'paired_diffdist':
        subject_id_col = getattr(runconfig, 'subject_id_col', 'subject_id')
        if aqutils.is_paired_eligible(runconfig.samplemap_df,
                                           condpair[0], condpair[1],
                                           subject_id_col=subject_id_col):
            paired_triples = aqutils.get_paired_sample_triples(
                runconfig.samplemap_df, condpair[0], condpair[1],
                subject_id_col=subject_id_col)
            use_paired = True
            LOGGER.info(f"paired_diffdist: contrast {condpair} is paired-eligible "
                        f"({len(paired_triples)} paired subjects); "
                        f"using DifferentialIonPaired")
            # Phase 2: build the paired empirical null ONCE per condpair,
            # before the per-ion loop. Pools within-(subject, arm) cell
            # log-FCs across all ions in each intensity bin.
            paired_normed = aqbg_paired.PairedConditionBackgrounds(
                df_c1_normed, df_c2_normed, paired_triples, p2z)
        else:
            LOGGER.warning(f"paired_diffdist requested but contrast {condpair} "
                           f"is not paired-eligible (no shared subjects in both "
                           f"arms via column '{subject_id_col}'); falling back "
                           f"to unpaired DifferentialIon for this contrast")

    bgpair2diffDist = {}
    deedpair2doublediffdist = {}
    count_ions=0
    all_diffions = []
    for ion in ions_to_check:
        vals1 = normed_c1.ion2nonNanvals.get(ion)
        vals2 = normed_c2.ion2nonNanvals.get(ion)
        bg1 = normed_c1.ion2background.get(ion)
        bg2 = normed_c2.ion2background.get(ion)
        diffDist = aqbg.get_subtracted_bg(bgpair2diffDist, bg1, bg2, p2z)
        if use_paired:
            # Phase 2: per-ion paired collapse — collapse multi-microdiss
            # (subject, arm) cell to a single scalar (mean of detected
            # log2-intensities). Drop subjects where either arm has no
            # detected values for this ion.
            paired_subject_pairs_for_ion = _build_paired_subject_pairs_for_ion(
                ion, paired_triples, df_c1_normed, df_c2_normed)
            paired_diff_dist = paired_normed.get_paired_diff_dist(ion)
            diffIon = aqdiff.DifferentialIonPaired(
                paired_subject_pairs=paired_subject_pairs_for_ion,
                paired_diff_dist=paired_diff_dist,
                name=ion,
                outlier_correction=runconfig.outlier_correction)
        elif runconfig.ion_test_method == 'ttest':
            diffIon = aqdiff.DifferentialIonTTest(vals1, vals2, ion, p2z, runconfig.outlier_correction)
        else:
            diffIon = aqdiff.DifferentialIon(vals1, vals2, diffDist, ion, runconfig.outlier_correction)
        protein = pep2prot.get(ion)
        if diffIon.usable:
            prot2diffions[protein].append(diffIon)
            all_diffions.append(diffIon)
        else:
            prot2missingval_diffions[protein].append(diffIon)



        if count_ions%2000==0:
            LOGGER.info(f"checked {count_ions} of {len(ions_to_check)} ions")

        count_ions+=1

    count_prots = 0
    for prot in prot2diffions.keys():
        ions = prot2diffions.get(prot)
        if len(ions)<runconfig.min_num_ions:
            continue



        clustered_prot_node = aqclust.get_scored_clusterselected_ions(prot, ions, normed_c1, normed_c2, bgpair2diffDist, p2z, deedpair2doublediffdist,
                                                                        pval_threshold_basis = runconfig.cluster_threshold_pval, fcfc_threshold = runconfig.cluster_threshold_fcfc,
                                                                        take_median_ion=runconfig.take_median_ion, fcdiff_cutoff_clustermerge= runconfig.fcdiff_cutoff_clustermerge,
                                                                        fragment_outlier_filtering=runconfig.fragment_outlier_filtering)
        protnodes.append(clustered_prot_node)

        if count_prots%100==0:
            LOGGER.info(f"checked {count_prots} of {len(prot2diffions.keys())} prots")

        count_prots+=1

    if len(prot2missingval_diffions.keys())>0:
        LOGGER.info(f"start analysis of proteins w. completely missing values")

        for prot in prot2missingval_diffions.keys():

            if prot in prot2diffions.keys(): #only do the missingval analysis if the protein was not analyzed the intensity-based way
                continue
            ions = prot2missingval_diffions.get(prot)
            protnode_missingval = aq_clust_missingval.create_protnode_from_missingval_ions(gene_name=prot,diffions=ions, normed_c1=normed_c1, normed_c2=normed_c2)
            if (protnode_missingval.c1_has_values) and (protnode_missingval.c2_has_values): #one of the conditions has to be missing, otherwise it means that there was e.g. one fragment ion with values in c1 and other fragment ions with values in c2
                continue
            protnodes_missingval.append(protnode_missingval)

        LOGGER.info(f"finished missing value analysis")

    if runconfig.use_ml:
        ml_performance_dict = {}

        #aq_class_stacked_frag.assign_predictability_scores_stacked(protein_nodes= protnodes, acquisition_info_df=None,results_dir=runconfig.results_dir, name = aqutils.get_condpairname(condpair)+"_fragions",
         #                           min_num_fragions=5, replace_nans=True, performance_metrics=ml_performance_dict, plot_predictor_performance=True)
        ml_successfull =aq_class_precursors.assign_predictability_scores(protein_nodes= protnodes, results_dir=runconfig.results_dir, name = aqutils.get_condpairname(condpair), ml_info_file=runconfig.ml_input_file,
                                        samples_used =c1_samples + c2_samples, min_num_precursors=3, prot_fc_cutoff=0, replace_nans=True, performance_metrics=ml_performance_dict, plot_predictor_performance=runconfig.runtime_plots)


        if ml_successfull and (ml_performance_dict["r2_score"] >0.05): #only use the ml score if it is meaningful
            aq_clust_mlreorder.update_nodes_w_ml_score(protnodes)
            LOGGER.info(f"ML based quality score above quality threshold and added to the nodes.")
            runconfig.ml_based_quality_score = True
        else:
            LOGGER.info(f"ML based quality score below quality threshold and not added to the nodes.")
            runconfig.ml_based_quality_score = False

    if runconfig.peptide_outlier_filtering:
        aq_clust_outlier.apply_peptide_outlier_filtering(protnodes)

    protnodes_combined = protnodes + protnodes_missingval

    # Phase 3b: permutation calibration for paired protein nodes.
    # MUST happen AFTER peptide_outlier_filtering and clustering have
    # finalized which leaves contribute to the protein-level test
    # (otherwise the permutation null wouldn't match the observed test).
    if use_paired:
        permutation_threshold = getattr(
            runconfig, 'paired_permutation_max_n_subjects',
            aqperm.MAX_N_FOR_FULL_ENUM)
        max_perms = getattr(
            runconfig, 'paired_permutation_max_perms',
            aqperm.DEFAULT_MAX_PERMS)
        _calibrate_protnodes_with_permutation(
            protnodes,
            permutation_threshold=permutation_threshold,
            max_perms=max_perms)

    condpair_node = aqclust_utils.get_condpair_node(protnodes_combined, condpair)
    condpair_node.fraction_missingval = len(prot2missingval_diffions.keys())/(len(prot2diffions.keys())+len(prot2missingval_diffions.keys()))
    condpair_node.is_paired_contrast = use_paired
    res_df, pep_df = write_out_tables(condpair_node, runconfig)

    LOGGER.info(f"condition pair {condpair} finished!")

    return res_df, pep_df


def _calibrate_protnodes_with_permutation(
    protnodes, permutation_threshold, max_perms,
):
    """Run permutation calibration on each paired protein node (Phase 3b).

    For each protein, recompute the gene-level z under each within-subject
    sign-flip permutation, then assign:

        node.p_val_analytic            = analytic Stouffer p (the
                                          existing node.p_val)
        node.p_val_permutation         = permutation-calibrated p
        node.permutation_p_floor       = 1 / N_perms
        node.n_perms_used              = N_perms
        node.correlation_estimator     = which estimator was used
        node.p_val                     = p_val_permutation when low n,
                                          else p_val_analytic

    Permutation strategy:
        - We approximate the multi-level tree aggregation by
          re-aggregating from the PROTEIN-LEVEL z-filtered children
          (typically peptide / mod_seq nodes whose z has been
          finalized by the clustering step) directly. The
          per-subject paired-fc dicts on each child carry the
          information needed to recompute z under a sign-flip.
        - z_child^perm = sign-weighted Stouffer-like sum over the
          child's per-subject z's (cached on the leaf via
          ``DifferentialIonPaired.paired_subject_zs`` and propagated up
          the tree by ``cluster_utils._merge_subject_fc_dicts_by_mean``
          equivalent for z).
        - Z_combined^perm = covariance-adjusted Stouffer of the
          permuted child z's, using the SAME R as the analytic test
          (R is invariant under within-subject sign flips because
          per-pair Cov(s_i·fc_i, s_i·fc_j) = s_i² Cov(fc_i, fc_j) =
          Cov(fc_i, fc_j)).

    Args:
        protnodes: list of gene-level anytree.Node from the analytic
            pass. Only nodes with `is_paired=True` are calibrated;
            others are skipped (they retain their analytic p_val).
        permutation_threshold: max n_subjects at which permutation is
            applied. Above this, only the analytic p is reported.
        max_perms: max permutations sampled at large n (default 64).

    Side effects:
        Mutates each paired protein node's attributes as listed above.
    """
    for prot in protnodes:
        if not getattr(prot, 'is_paired', False):
            continue
        # Establish the analytic baseline.
        prot.p_val_analytic = float(prot.p_val)
        prot.correlation_estimator = 'ledoit_wolf'

        # The subjects available at the protein level.
        prot_fcs = getattr(prot, 'paired_subject_fcs', None) or {}
        n_subj = len(prot_fcs)
        if n_subj <= 1:
            prot.p_val_permutation = float(prot.p_val_analytic)
            prot.permutation_p_floor = 1.0
            prot.n_perms_used = 0
            continue
        if n_subj > permutation_threshold:
            # High-n: skip permutation (analytic asymptotics fine).
            prot.p_val_permutation = float(prot.p_val_analytic)
            prot.permutation_p_floor = 1.0 / aqperm.compute_n_perms_for_n_subjects(
                n_subj, max_perms=max_perms)
            prot.n_perms_used = 0
            continue

        # Pull the leaves that contributed to the analytic test for the
        # protein. ``obtain_all_included_leaves`` walks the tree and
        # returns only leaves where every ancestor (above the root) has
        # ``is_included=True`` and ``cluster==0`` — exactly the
        # mainclust-leaf set that
        # ``cluster_along_specified_levels`` aggregated when computing
        # the protein-level z. Additionally drop leaves whose peptide
        # ancestor was flagged ``is_outlier_peptide=True`` (that flag is
        # honoured by ``get_selected_nodes_for_zvalcalc`` at the gene
        # level but not by ``obtain_all_included_leaves``).
        def _peptide_ancestor_is_outlier(leaf_node):
            n = leaf_node.parent
            while n is not None:
                if getattr(n, 'type', None) == 'seq':
                    return bool(getattr(n, 'is_outlier_peptide', False))
                n = n.parent
            return False

        leaves = [
            leaf for leaf in aqclust_utils.obtain_all_included_leaves(prot)
            if getattr(leaf, 'is_paired', False)
            and not _peptide_ancestor_is_outlier(leaf)
        ]
        if len(leaves) == 0:
            prot.p_val_permutation = float(prot.p_val_analytic)
            prot.permutation_p_floor = 1.0
            prot.n_perms_used = 0
            continue

        # Establish a canonical subject ordering across the protein (the
        # union of subjects from any contributing leaf).
        subject_set: set = set()
        for leaf in leaves:
            d = getattr(leaf, 'paired_subject_zs', None) or {}
            subject_set.update(d.keys())
        subject_order = sorted(subject_set)
        n_subj_perm = len(subject_order)
        if n_subj_perm <= 1:
            prot.p_val_permutation = float(prot.p_val_analytic)
            prot.permutation_p_floor = 1.0
            prot.n_perms_used = 0
            continue

        # Per-leaf {subject: z} matrix shape (n_subj_perm, n_leaves).
        # NaN where a subject is missing for a leaf.
        Z = np.full((n_subj_perm, len(leaves)), np.nan, dtype=float)
        for j, leaf in enumerate(leaves):
            z_dict = getattr(leaf, 'paired_subject_zs', None) or {}
            for i, subj in enumerate(subject_order):
                if subj in z_dict:
                    Z[i, j] = float(z_dict[subj])

        # Per-leaf {subject: fc} matrix for correlation estimation
        # (re-using the analytic R cached on the protein node would be
        # ideal, but it was built over the protein's z-filtered
        # IMMEDIATE children — at the gene level those are peptides,
        # not leaves. For the permutation calibrator we work at leaf
        # granularity, so we compute the leaf-level R fresh here.)
        FC = np.full((n_subj_perm, len(leaves)), np.nan, dtype=float)
        for j, leaf in enumerate(leaves):
            fc_dict = getattr(leaf, 'paired_subject_fcs', None) or {}
            for i, subj in enumerate(subject_order):
                if subj in fc_dict:
                    FC[i, j] = float(fc_dict[subj])
        R_leaf = aq_corr.compute_ion_correlation_matrix(
            FC, method='ledoit_wolf')

        # Per-leaf summed-z normalization factor: under the analytic
        # path, leaf z = sum(z_i) / sqrt(N_i_subjects). Replicate that
        # for the permutation null so the permutation z's are on the
        # same scale as the observed leaf z's.
        # Mask of non-NaN per leaf (subject availability per leaf).
        mask = np.isfinite(Z)
        n_subj_per_leaf = np.sum(mask, axis=0)  # length n_leaves
        # Replace NaNs with 0 in Z so flipping/sums work cleanly (NaN
        # rows simply contribute 0).
        Z_filled = np.where(mask, Z, 0.0)
        sqrt_n = np.where(n_subj_per_leaf > 0, np.sqrt(n_subj_per_leaf), 1.0)

        perms = aqperm.enumerate_signflip_permutations(
            n_subj_perm, max_perms=max_perms,
            include_identity=True)
        n_perms = len(perms)
        if n_perms == 0:
            prot.p_val_permutation = float(prot.p_val_analytic)
            prot.permutation_p_floor = 1.0
            prot.n_perms_used = 0
            continue

        # Permutation z distribution at the protein level.
        perm_zs = np.zeros(n_perms, dtype=float)
        for k, signs in enumerate(perms):
            # Apply sign flip per subject, then sum / sqrt(N_subj_per_leaf)
            # to get permuted leaf z's. NaN-masked entries contribute 0.
            Z_perm = Z_filled * signs[:, None]
            leaf_z_sums = np.sum(Z_perm, axis=0)
            leaf_z = leaf_z_sums / sqrt_n
            # Covariance-adjusted Stouffer at protein level.
            z_combined = aq_corr.covariance_adjusted_stouffer_z(
                leaf_z, R_leaf)
            perm_zs[k] = z_combined

        # Observed gene-level z (analytic, full-tree aggregation).
        observed_z = float(prot.z_val) if prot.z_val is not None else 0.0
        p_perm, p_floor = aqperm.permutation_calibrated_p(observed_z, perm_zs)

        prot.p_val_permutation = p_perm
        prot.permutation_p_floor = p_floor
        prot.n_perms_used = n_perms
        # Default p_val swaps to the permutation-calibrated value when
        # we actually ran permutations (low-n regime). Documents the
        # honest p that respects the discrete floor.
        prot.p_val = p_perm

import alphaquant.diffquant.diffutils as aqutils
def get_unnormed_df_condpair(input_file:str, samplemap_df:pd.DataFrame, condpair:str, file_has_alphaquant_format: bool) -> pd.DataFrame:


    samples_c1, samples_c2 = aqutils.get_samples_used_from_samplemap_df(samplemap_df=samplemap_df, cond1 = condpair[0], cond2 = condpair[1])
    used_samples = samples_c1+samples_c2
    unnormed_df = aqutils.import_data(input_file, samples_subset=used_samples, file_has_alphaquant_format = file_has_alphaquant_format)
    unnormed_df, _ = aqutils.prepare_loaded_tables(unnormed_df, samplemap_df)
    return unnormed_df



def write_out_normed_df(normed_df_1, normed_df_2, pep2prot, results_dir, condpair):
    merged_df = normed_df_1.merge(normed_df_2, left_index = True, right_index = True)
    merged_df = 2**merged_df
    merged_df = merged_df.replace(np.nan, 0)
    merged_df["protein"] = list(map(lambda x : pep2prot.get(x),merged_df.index))
    if not os.path.exists(f"{results_dir}/"):
        os.makedirs(f"{results_dir}/")
    merged_df.to_csv(f"{results_dir}/{aqutils.get_condpairname(condpair)}.normed.tsv", sep = "\t")


def get_per_condition_dataframes(samples_c1, samples_c2, unnormed_df, min_valid_values, valid_values_filter_mode, min_valid_values_c1, min_valid_values_c2):

    min_samples = min(len(samples_c1), len(samples_c2))

    if min_samples<2:
        raise Exception(f"condpair has not enough samples: c1:{len(samples_c1)} c2: {len(samples_c2)}, skipping")

    if valid_values_filter_mode == "either":
        min_valid_values = np.min([get_min_valid_values_for_cond(samples_c1, min_valid_values), get_min_valid_values_for_cond(samples_c2, min_valid_values)])
        passes_min_valid_values_c1 = unnormed_df.loc[:, samples_c1].notna().sum(axis=1) >= min_valid_values
        passes_min_valid_values_c2 = unnormed_df.loc[:, samples_c2].notna().sum(axis=1) >= min_valid_values
        passes_min_valid_values = passes_min_valid_values_c1 | passes_min_valid_values_c2
        unnormed_df = unnormed_df[passes_min_valid_values]
        df_c1 = unnormed_df.loc[:, samples_c1]
        df_c2 = unnormed_df.loc[:, samples_c2]

    elif valid_values_filter_mode == "both":
        min_valid_values_c1 = get_min_valid_values_for_cond(samples_c1, min_valid_values)
        min_valid_values_c2 = get_min_valid_values_for_cond(samples_c2, min_valid_values)
        df_c1 = unnormed_df.loc[:, samples_c1].dropna(thresh=min_valid_values_c1, axis=0)
        df_c2 = unnormed_df.loc[:, samples_c2].dropna(thresh=min_valid_values_c2, axis=0)

    elif valid_values_filter_mode == "per_condition":
        min_valid_values_c1 = get_min_valid_values_for_cond(samples_c1, min_valid_values_c1)
        min_valid_values_c2 = get_min_valid_values_for_cond(samples_c2, min_valid_values_c2)
        df_c1 = unnormed_df.loc[:, samples_c1].dropna(thresh=min_valid_values_c1, axis=0)
        df_c2 = unnormed_df.loc[:, samples_c2].dropna(thresh=min_valid_values_c2, axis=0)
    else:
        raise Exception(f"invalid value set for the variable valid_values_filter_mode: {valid_values_filter_mode}, please ensure that is set to: 'either', 'both' or 'per_condition'")

    if (len(df_c1.index)<5) | (len(df_c2.index)<5):
        raise Exception(f"condpair has not enough data for processing c1: {len(df_c1.index)} c2: {len(df_c2.index)}, skipping")

    return df_c1, df_c2

def get_min_valid_values_for_cond(c_samples, min_valid_values):
    if min_valid_values is None: #in the case of None, no nans will be allowed
        return None
    num_samples = len(c_samples)
    if num_samples<min_valid_values:
        return num_samples
    else:
        return min_valid_values




def write_out_tables(condpair_node, runconfig):
    condpair = condpair_node.name

    res_df = aq_tablewriter_protein.TableFromNodeCreator(condpair_node, node_type = "gene", min_num_peptides = runconfig.minpep, annotation_file= getattr(runconfig, "annotation_file", None)).results_df
    has_sequence_nodes = check_if_has_sequence_nodes(condpair_node)
    if has_sequence_nodes:
        pep_df = aq_tablewriter_protein.TableFromNodeCreator(condpair_node, node_type = "seq").results_df
    else:
        pep_df = None
    has_precursor_nodes = check_if_has_precursor_nodes(condpair_node)
    if has_precursor_nodes:
        prec_df = aq_tablewriter_protein.TableFromNodeCreator(condpair_node, node_type = "mod_seq_charge").results_df
    else:
        prec_df = None

    has_base_nodes = check_if_has_base_nodes(condpair_node)
    if has_base_nodes and runconfig.write_base_ions:
        base_df = aq_tablewriter_protein.TableFromNodeCreator(condpair_node, node_type = "base").results_df
    else:
        base_df = None


    if runconfig.runtime_plots:
        aq_plot_pairwise.volcano_plot(res_df, fdr_cutoff= runconfig.volcano_fdr, log2fc_cutoff = runconfig.volcano_fcthresh)
        if has_sequence_nodes:
            aq_plot_pairwise.volcano_plot(pep_df,fdr_cutoff = runconfig.volcano_fdr, log2fc_cutoff = runconfig.volcano_fcthresh)

    if runconfig.results_dir!=None:
        if runconfig.write_out_results_tree:
            aqclust_utils.export_condpairtree_to_json(condpair_node, results_dir = runconfig.results_dir)
        # Write proteoform table defensively; skip if errors occur
        try:
            proteoform_df = aq_tablewriter_proteoform.ProteoFormTableCreator(
                condpair_tree=condpair_node, organism=runconfig.organism
            ).proteoform_df
            proteoform_df.to_csv(
                f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.proteoforms.tsv",
                sep='\t', index=False
            )
        except Exception as e:
            LOGGER.warning(f"Skipping proteoform table write due to error: {e}")

        runconfig_df = aq_tablewriter_runconfig.RunConfigTableCreator(runconfig).runconfig_df

        runconfig_df.to_csv(f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.runconfig.tsv", sep='\t', header=False)
        res_df.to_csv(f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.results.tsv", sep = "\t", index=None)
        if has_sequence_nodes:
            pep_df.to_csv(f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.results.seq.tsv", sep = "\t", index=None)

        if has_precursor_nodes:
            prec_df.to_csv(f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.results.prec.tsv", sep = "\t", index=None)

        if base_df is not None:
            base_df.to_csv(f"{runconfig.results_dir}/{aqutils.get_condpairname(condpair)}.results.base.tsv", sep = "\t", index=None)

    return res_df, pep_df

def check_if_has_sequence_nodes(condpair_node):
    return condpair_node.children[0].children[0].type == "seq"

def check_if_has_precursor_nodes(condpair_node):
    try:
        return condpair_node.children[0].children[0].children[0].children[0].type == "mod_seq_charge"
    except:
        return False

def check_if_has_base_nodes(condpair_node):
    try:
        # Check if we have base nodes (fragments/MS1) at the leaf level
        return condpair_node.children[0].leaves[0].type == "base"
    except:
        return False


def _build_paired_subject_pairs_for_ion(ion, paired_triples, df_c1_normed, df_c2_normed):
    """Per-ion paired collapse for the paired-diffdist test (Phase 2).

    For each (subject, c1_samples, c2_samples) triple, take the mean of the
    detected log2 intensities in each arm for this ion. Drop subjects where
    either arm has no detected values for this ion (purely defensive — the
    DifferentialIonPaired constructor checks ``len >= 2`` independently;
    keeping zero-detect subjects would inject NaNs into the test).

    Args:
        ion: ion identifier (must be present in both df_c1_normed and
            df_c2_normed indices, otherwise this ion would not be in
            ions_to_check; defensive lookup uses ``.loc.get`` semantics
            via membership check).
        paired_triples: from
            :func:`aqutils.get_paired_sample_triples` — list of
            ``(subject_id, c1_samples, c2_samples)`` tuples.
        df_c1_normed, df_c2_normed: wide-format (ions × samples) log2
            dataframes, ONE arm per dataframe.

    Returns:
        list of ``(subject_id, val_arm1, val_arm2)`` tuples, one per
        subject with detected values in BOTH arms. Subjects with all NaN
        in either arm are dropped. Empty list if no subject is fully
        observed for this ion.
    """
    # Cheap presence check: if the ion is absent from either frame's index,
    # there's nothing to collapse. (ions_to_check is the index intersection
    # so this should be rare, but guard anyway.)
    if ion not in df_c1_normed.index or ion not in df_c2_normed.index:
        return []

    # Pull the row once per arm to avoid N df.loc lookups inside the loop.
    row_c1 = df_c1_normed.loc[ion]
    row_c2 = df_c2_normed.loc[ion]

    pairs = []
    for subj, c1_samples, c2_samples in paired_triples:
        # Filter to columns that actually exist in the per-arm frame.
        c1_present = [s for s in c1_samples if s in row_c1.index]
        c2_present = [s for s in c2_samples if s in row_c2.index]
        if not c1_present or not c2_present:
            continue
        c1_vals = row_c1.loc[c1_present].to_numpy(dtype=float)
        c2_vals = row_c2.loc[c2_present].to_numpy(dtype=float)
        c1_vals = c1_vals[~np.isnan(c1_vals)]
        c2_vals = c2_vals[~np.isnan(c2_vals)]
        if c1_vals.size == 0 or c2_vals.size == 0:
            continue
        pairs.append((subj, float(np.mean(c1_vals)), float(np.mean(c2_vals))))
    return pairs
