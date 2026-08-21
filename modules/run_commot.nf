// Single per-region COMMOT process. NOTE: the Explorer sbatch runs a 24-way
// pair-chunked array (run_commot.py --pair-chunk i/N -> merge_commot.py) for the
// large GW20 inputs; for portability the Nextflow model runs one process per
// region (raise cpu_large resources / re-introduce chunking for very large inputs).
process RUN_COMMOT {
    label 'cpu_large'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/commot", mode: 'copy'

    input:
    tuple val(region), path(envi_h5ad)

    output:
    tuple val(region), path("commot_${region}.h5ad"), emit: h5ad

    script:
    """
    python ${projectDir}/scripts/commot/run_commot.py \
        --region ${region} \
        --h5ad ${envi_h5ad} \
        --layers ${params.layers} \
        --dis-thr ${params.dis_thr} \
        --cot-nitermax ${params.cot_nitermax} \
        --outdir .
    """

    stub:
    "touch commot_${region}.h5ad"
}
