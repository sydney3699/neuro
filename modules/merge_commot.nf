// merge_commot reassembles pair-chunked COMMOT output. In the single-process
// Nextflow model RUN_COMMOT already emits a full commot_<region>.h5ad, so this
// process is only needed if the chunked variant is reintroduced. Kept for parity.
process MERGE_COMMOT {
    label 'cpu_large'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/commot", mode: 'copy'

    input:
    tuple val(region), path(chunk_dir), path(envi_h5ad)

    output:
    tuple val(region), path("commot_${region}.h5ad"), emit: h5ad

    script:
    """
    python ${projectDir}/scripts/commot/merge_commot.py \
        --region ${region} --chunk-dir ${chunk_dir} \
        --h5ad ${envi_h5ad} --layers ${params.layers} --outdir .
    """

    stub:
    "touch commot_${region}.h5ad"
}
