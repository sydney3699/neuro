process BUILD_GW34_INPUT {
    label 'cpu_small'
    container "${params.registry}/neuro-analysis:${params.tag}"
    publishDir "${params.outdir}/data", mode: 'copy'

    input:
    path raw_h5ad

    output:
    path "gw34_ba17_v1v2_envi_input.h5ad", emit: h5ad

    script:
    """
    python ${projectDir}/scripts/envi/build_gw34_envi_input.py \
        --in-h5ad ${raw_h5ad} \
        --out-h5ad gw34_ba17_v1v2_envi_input.h5ad \
        --areas ${params.areas}
    """

    stub:
    "touch gw34_ba17_v1v2_envi_input.h5ad"
}
