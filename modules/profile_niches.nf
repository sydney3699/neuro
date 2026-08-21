process PROFILE_NICHES {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir { "${params.outdir}/niche_${region}_${method}K${params.k}_${annot}" }, mode: 'copy'

    input:
    // one (region, method, annotation) combo on the shared commot substrate
    tuple val(region), val(method), val(annot), path(commot_h5ad), path(domain_parquet), path(annot_parquet)

    output:
    tuple val(region), val(method), val(annot), path("${region}_niche_*.csv"), path("${region}_boundary_cells.parquet"), emit: profile

    script:
    """
    python ${projectDir}/scripts/niches/profile_niches.py \
        --h5ad ${commot_h5ad} --region ${region} \
        --domain-parquet ${domain_parquet} --domain-key domain_k${params.k} \
        --annotation-parquet ${annot_parquet} --annotation-col annotation \
        --outdir .
    """

    stub:
    """
    touch ${region}_niche_composition.csv ${region}_niche_signaling.csv \
          ${region}_niche_profile.csv ${region}_boundary_cells.parquet
    """
}
