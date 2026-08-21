process ANNOTATE_SCVI {
    label 'gpu'
    container "${params.registry}/neuro-scvi:${params.tag}"
    publishDir "${params.outdir}/annot_scvi", mode: 'copy'

    input:
    path spatial_h5ad
    path snrna_ref

    output:
    tuple val('scvi'), path("${params.tag_name}_scvi_annotation.parquet"), emit: annotation
    path "${params.tag_name}_scvi_cluster_map.csv"

    script:
    """
    python ${projectDir}/scripts/annotation/annotate_cells.py \
        --spatial-h5ad ${spatial_h5ad} --reference-h5ad ${snrna_ref} \
        --embedding scvi --ref-label-key ${params.ref_label_key} \
        --leiden-resolution ${params.leiden_resolution} \
        --tag ${params.tag_name} --outdir .
    """

    stub:
    "touch ${params.tag_name}_scvi_annotation.parquet ${params.tag_name}_scvi_cluster_map.csv"
}

process SCGPT_EMBED {
    label 'gpu'
    container "${params.registry}/neuro-scgpt:${params.tag}"
    publishDir "${params.outdir}/scgpt_embed", mode: 'copy'

    input:
    path spatial_h5ad

    output:
    path "${params.tag_name}_scgpt_embedding.parquet", emit: embedding

    script:
    """
    python ${projectDir}/scripts/annotation/scgpt_embed.py \
        --spatial-h5ad ${spatial_h5ad} --model-dir ${params.scgpt_model_dir} \
        --tag ${params.tag_name} --outdir .
    """

    stub:
    "touch ${params.tag_name}_scgpt_embedding.parquet"
}

process ANNOTATE_SCGPT {
    label 'cpu_small'
    container "${params.registry}/neuro-scvi:${params.tag}"
    publishDir "${params.outdir}/annot_scgpt", mode: 'copy'

    input:
    path spatial_h5ad
    path snrna_ref
    path scgpt_parquet

    output:
    tuple val('scgpt'), path("${params.tag_name}_scgpt_annotation.parquet"), emit: annotation
    path "${params.tag_name}_scgpt_cluster_map.csv"

    script:
    """
    python ${projectDir}/scripts/annotation/annotate_cells.py \
        --spatial-h5ad ${spatial_h5ad} --reference-h5ad ${snrna_ref} \
        --embedding-parquet ${scgpt_parquet} --emb-name scgpt \
        --ref-label-key ${params.ref_label_key} \
        --leiden-resolution ${params.leiden_resolution} \
        --tag ${params.tag_name} --outdir .
    """

    stub:
    "touch ${params.tag_name}_scgpt_annotation.parquet ${params.tag_name}_scgpt_cluster_map.csv"
}
