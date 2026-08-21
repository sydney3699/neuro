process BANKSY_DOMAINS {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/banksy", mode: 'copy'

    input:
    tuple val(region), path(envi_h5ad)

    output:
    tuple val(region), val('banksy'), path("${region}_banksy_domains.parquet"), emit: domains

    script:
    """
    python ${projectDir}/scripts/domains/banksy_domains.py \
        --region ${region} --h5ad ${envi_h5ad} \
        --layers ${params.layers} --kmin ${params.kmin} --kmax ${params.kmax} \
        --outdir .
    """

    stub:
    "touch ${region}_banksy_domains.parquet"
}

process STAGATE_DOMAINS {
    label 'gpu'
    container "${params.registry}/neuro-stagate:${params.tag}"
    publishDir "${params.outdir}/stagate", mode: 'copy'

    input:
    tuple val(region), path(envi_h5ad)

    output:
    tuple val(region), val('stagate'), path("${region}_stagate_domains.parquet"), emit: domains

    script:
    """
    python ${projectDir}/scripts/domains/stagate_domains.py \
        --region ${region} --h5ad ${envi_h5ad} \
        --layers ${params.layers} --kmin ${params.kmin} --kmax ${params.kmax} \
        --outdir .
    """

    stub:
    "touch ${region}_stagate_domains.parquet"
}
