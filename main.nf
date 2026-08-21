#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { RUN_ENVI }                      from './modules/run_envi.nf'
include { BUILD_GW34_INPUT }             from './modules/build_gw34_input.nf'
include { RUN_COMMOT }                    from './modules/run_commot.nf'
include { BANKSY_DOMAINS; STAGATE_DOMAINS } from './modules/domains.nf'
include { ANNOTATE_SCVI; SCGPT_EMBED; ANNOTATE_SCGPT } from './modules/annotate.nf'
include { PROFILE_NICHES }               from './modules/profile_niches.nf'
include { COMPARE_NICHES; SIGNALING_DIFF; SP_NICHE; LAYER_RECOVERY } from './modules/compare.nf'
include { HARMONIZE_H1; PERSISTENCE_COMPARE } from './modules/gw34.nf'

// ---------------------------------------------------------------------------
// GW20 2x2 comparison: {scVI,scGPT} annotation x {Banksy,STAGATE} domains on one
// shared ENVI->COMMOT substrate, then the cross-method/region/annotation contrasts.
// ---------------------------------------------------------------------------
workflow GW20_2X2 {
    spatial = file(params.gw20_input)
    snrna   = file(params.snrna_ref)
    regions = Channel.fromList(params.regions.tokenize(','))

    RUN_ENVI(spatial, snrna)
    envi = RUN_ENVI.out.h5ad

    // substrate: COMMOT + both domain methods, per region
    commot  = RUN_COMMOT(regions.combine(envi))               // (region, commot_h5ad)
    banksy  = BANKSY_DOMAINS(regions.combine(envi))           // (region, 'banksy', parquet)
    stagate = STAGATE_DOMAINS(regions.combine(envi))          // (region, 'stagate', parquet)
    domains = banksy.domains.mix(stagate.domains)

    // annotation arms (shared across the substrate)
    scvi  = ANNOTATE_SCVI(spatial, snrna)                     // ('scvi', parquet)
    emb   = SCGPT_EMBED(spatial)
    scgpt = ANNOTATE_SCGPT(spatial, snrna, emb.embedding)     // ('scgpt', parquet)
    annots = scvi.annotation.mix(scgpt.annotation)

    // profile every (region, method, annotation) combo on the commot substrate
    combos = domains
        .combine(commot.h5ad, by: 0)                          // (region, method, dom_pq, commot_h5ad)
        .combine(annots)                                      // (region, method, dom_pq, commot_h5ad, annot, annot_pq)
        .map { region, method, dom_pq, commot_h5ad, annot, annot_pq ->
               tuple(region, method, annot, commot_h5ad, dom_pq, annot_pq) }
    PROFILE_NICHES(combos)

    // fan-in comparison stages. They read the published results tree via
    // ${params.outdir}; the collected upstream channel is the dependency gate.
    ready = PROFILE_NICHES.out.profile.map { it[3] }.collect()
    COMPARE_NICHES(ready)
    SP_NICHE(ready)
    LAYER_RECOVERY(ready)
    SIGNALING_DIFF(commot.h5ad.map { it[1] }.collect(), "${params.outdir}/commot")
}

// ---------------------------------------------------------------------------
// GW34 persistence: mirror the substrate on GW34, build niches at native H1,
// then compare V1-vs-V2 signaling/composition/niches against GW20.
// ---------------------------------------------------------------------------
workflow GW34_PERSISTENCE {
    raw     = file(params.gw34_raw)
    snrna   = file(params.snrna_ref)
    regions = Channel.fromList(params.regions.tokenize(','))

    BUILD_GW34_INPUT(raw)
    RUN_ENVI(BUILD_GW34_INPUT.out.h5ad, snrna)
    envi = RUN_ENVI.out.h5ad

    commot  = RUN_COMMOT(regions.combine(envi))
    stagate = STAGATE_DOMAINS(regions.combine(envi))

    // GW34 uses native H1 labels (built by harmonize_h1) as the annotation parquet
    HARMONIZE_H1(params.gw20_commot_dir, commot.h5ad.map { it[1] }.collect())
    gw34_annot = HARMONIZE_H1.out.parquet.flatten().filter { it.name.startsWith('gw34_') }

    combos = stagate.domains
        .combine(commot.h5ad, by: 0)
        .map { region, method, dom_pq, commot_h5ad ->
               tuple(region, method, 'h1', commot_h5ad, dom_pq, file(params.gw34_h1_placeholder)) }
    PROFILE_NICHES(combos)

    SIGNALING_DIFF(commot.h5ad.map { it[1] }.collect(), "${params.outdir}/commot")
    persist_ready = PROFILE_NICHES.out.profile.map { it[3] }
        .mix(SIGNALING_DIFF.out.csv, HARMONIZE_H1.out.parquet).collect()
    PERSISTENCE_COMPARE(persist_ready)
}

workflow {
    if( params.entry == 'gw34' ) { GW34_PERSISTENCE() }
    else                         { GW20_2X2() }
}
