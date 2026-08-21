// Fan-in comparison stages. They read the PUBLISHED results tree via ${params.outdir}
// (populated by publishDir upstream); the `val ready` input is only a dependency
// gate (a collected list of upstream outputs) so they run after profiling completes.
process COMPARE_NICHES {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/niche_comparison", mode: 'copy'

    input:
    val ready

    output:
    path "*.csv", emit: csv

    script:
    """
    python ${projectDir}/scripts/niches/compare_niches.py \
        --results ${params.outdir} \
        --annotations ${params.annotations} --methods ${params.methods} \
        --regions ${params.regions} --k ${params.k} \
        --kmin ${params.kmin} --kmax ${params.kmax} \
        --n-perm ${params.n_perm} --outdir .
    """

    stub:
    "touch crossmethod_ari_by_k.csv crossregion_match_by_k.csv annotation_axis_cosine_by_k.csv"
}

process SIGNALING_DIFF {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/signaling_diff", mode: 'copy'

    input:
    val ready
    val commot_dir

    output:
    path "*.csv", emit: csv

    script:
    """
    python ${projectDir}/scripts/niches/signaling_diff.py \
        --commot-dir ${commot_dir} --regions ${params.regions} \
        --level pathway --outdir .
    """

    stub:
    "touch signaling_diff_cpmz.csv wnt_recheck_summary.csv"
}

process SP_NICHE {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/sp_niche", mode: 'copy'

    input:
    val ready

    output:
    path "*.csv", emit: csv

    script:
    """
    python ${projectDir}/scripts/niches/sp_niche_analysis.py \
        --results ${params.outdir} --regions ${params.regions} \
        --methods ${params.methods} --annotations ${params.annotations} \
        --kmin ${params.kmin} --kmax ${params.kmax} --outdir .
    """

    stub:
    "touch sp_niche_summary.csv sp_cell_direct_summary.csv"
}

process LAYER_RECOVERY {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/layer_recovery", mode: 'copy'

    input:
    val ready

    output:
    path "layer_recovery_by_k.csv", emit: csv

    script:
    """
    python ${projectDir}/scripts/niches/layer_recovery_eval.py \
        --results ${params.outdir} --regions ${params.regions} \
        --methods ${params.methods} --kmin ${params.kmin} --kmax ${params.kmax} \
        --outdir .
    """

    stub:
    "touch layer_recovery_by_k.csv"
}
