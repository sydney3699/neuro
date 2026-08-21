process RUN_ENVI {
    label 'gpu'
    container "${params.registry}/neuro-envi:${params.tag}"
    publishDir "${params.outdir}/envi", mode: 'copy'

    input:
    path spatial_h5ad
    path snrna_ref

    output:
    path "envi_out/FB080_spatial_envi.h5ad", emit: h5ad
    path "envi_out", emit: dir

    script:
    """
    python ${projectDir}/scripts/envi/run_envi.py \
        --spatial ${spatial_h5ad} \
        --snrna ${snrna_ref} \
        --outdir envi_out \
        --training-steps ${params.envi_training_steps} \
        --num-hvg ${params.envi_num_hvg} \
        --lr-database ${params.lr_database} \
        --celltype-key ${params.envi_celltype_key}
    """

    stub:
    "mkdir -p envi_out && touch envi_out/FB080_spatial_envi.h5ad"
}
