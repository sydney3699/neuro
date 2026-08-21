process HARMONIZE_H1 {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/h1harm", mode: 'copy'

    input:
    val gw20_commot_dir
    val ready              // gate: GW34 commot done

    output:
    path "*.parquet", emit: parquet
    path "h1harm_class_counts.csv"

    script:
    """
    python ${projectDir}/scripts/gw34/harmonize_h1.py \
        --gw20-commot-dir ${gw20_commot_dir} \
        --gw34-commot-dir ${params.outdir}/commot \
        --regions ${params.regions} --outdir .
    """

    stub:
    """
    touch gw20_v1_h1harm.parquet gw20_v2_h1harm.parquet \
          gw34_v1_h1harm.parquet gw34_v2_h1harm.parquet h1harm_class_counts.csv
    """
}

process PERSISTENCE_COMPARE {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/persistence", mode: 'copy'

    input:
    val ready              // gate: signaling_diff + profiling + harmonize done

    output:
    path "persistence_*.csv", emit: csv

    script:
    """
    python ${projectDir}/scripts/gw34/persistence_compare.py \
        --gw20-signaling ${params.gw20_signaling} \
        --gw34-signaling ${params.outdir}/signaling_diff/signaling_diff_cpmz.csv \
        --harm-dir ${params.outdir}/h1harm \
        --gw20-niche-dir ${params.gw20_niche_dir} \
        --gw34-niche-dir ${params.outdir}/niche_gw34_stagate_K8 \
        --regions ${params.regions} --outdir .
    """

    stub:
    "touch persistence_signaling.csv persistence_composition.csv persistence_niches.csv"
}
