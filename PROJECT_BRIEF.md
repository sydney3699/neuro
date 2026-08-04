# Project Brief

**Title:** Spatially-resolved cell-cell communication and the V1/V2 boundary in the developing human cortex: a multi-method comparative analysis

**Author:** Sydney Cole

**Started:** 2026-05-21

**Target completion:** 

**Last updated:** 2026-08-03

---

## Summary

This project analyzes spatially-resolved cell-cell communication (CCC) at the V1/V2 boundary in the developing human cortex, using the Qian/Walsh et al. 2025 MERFISH atlas. It extends the paper's central finding (sharp molecular V1/V2 boundary at GW20) by asking what signaling environments distinguish the two areas, and whether these signaling differences persist into late gestation (GW34). The project simultaneously evaluates whether modern deep learning methods (single-cell foundation models for annotation; graph neural networks for spatial domain identification) change which signaling niches are identified compared to standard analytical pipelines.

---

## Biological question

**Primary:** What spatially-resolved cell-cell communication niches distinguish V1 (BA17) from V2 (BA18) in the developing human cortex at GW20, and do these signaling niches persist at GW34?

**Why this matters:** The Qian/Walsh paper established that the V1/V2 boundary is molecularly sharp at GW20, well before cytoarchitectural differentiation. The paper proposes that synaptogenesis in V1-specific L4 neurons contributes to this boundary but does not identify the broader signaling environment that supports area-specific molecular identity. Spatial CCC analysis can address this gap by identifying ligand-receptor interactions enriched in V1 versus V2 cortical plate, contributing to mechanistic understanding of how a discrete molecular boundary is established and maintained in tissue otherwise governed by gradient-like specification.

**Secondary:** Within EN-ET cells in the cortical plate at GW20, do the three subplate (SP) populations (EN-ET-SP, EN-ET-SP-P, EN-ET-SP-early) differ in their local signaling environments between V1 and V2? 

**Extension after initial results:** Using best-performing pipeline, do the same GW20 V1-vs-V2 niches persist at GW34?

---

## Methodological question

Do single-cell foundation model embeddings (for cell type annotation) and graph neural network spatial methods (for spatial domain identification) change which V1/V2 signaling niches are identified compared to standard pipelines? Specifically:

1. Do foundation model embeddings more accurately distinguish V1-specific from V2-specific subpopulations within layer-fated EN-IT classes?
2. Do GNN-based spatial methods more accurately recover the known sharp V1/V2 boundary (using the paper's area labels as ground truth) than locally-aware spatial clustering?
3. Do these methodological choices propagate to differences in inferred CCC niches?

The 2x2 comparison structure is detailed in the analytical pipeline section below.

---

## Data

### Primary datasets
- **MERFISH master integrated atlas:** `merscope_integrated_855.h5ad` (14.2 GB, 15.9M cells). Qian/Walsh et al. 2025, *Nature*. Zenodo: [10.5281/zenodo.14422018](https://doi.org/10.5281/zenodo.14422018) (CC-BY 4.0).
- Used for GW20 analyses; subset to V1/V2 areas (A-V1, B-V1, A-V2, B-V2) and cortical plate + marginal zone layers.
- **Paired snRNA-seq:** 91,898 cells from the same samples as MERFISH. Used for 
  ENVI imputation of whole-transcriptome expression. Required for L-R-based 
  CCC analysis. Location: investigate Final_Integration_MDM_100323.rds or 
  contact authors.

- **ENVI-imputed expression:** Whole-transcriptome (or top 1,000 HVGs) imputed 
  from snRNA-seq onto MERFISH cells. May be available as paper artifact, 
  otherwise generated as part of project pipeline.

### Secondary dataset
- **Per-region BA17 file:** `gw34_umb5900_ba17.h5ad` (309 MB, 240K cells; 38.5K with area assignments).
- Used for GW34 V1/V2 comparison. Contains both A-V1 (25.8K cells) and B-V2 (12.8K cells) at GW34 from UMB5900.
- Required because the master file does not retain GW34 BA17 V1 cells.

### Reference and validation datasets
- **Paired snRNA-seq (normalized):** `norm_exp.h5ad` (7.4 GB, 5.7M cells). Used for cell type annotation transfer and ENVI-imputed expression validation.
- **Visium (single section):** `Visium_A1_brain_011124.rds`. Whole-transcriptome validation of identified L-R interactions on one section.
- **External developmental reference:** Braun et al. 2023 (CELLxGENE collection 4d8fed08). Used for foundation model embedding comparison and cross-dataset annotation validation.

### Cell type annotations used
- **GW20 (master file):** Full H1/H2 taxonomy; harmonized to UL/DL grouping for cross-time-point comparison
- **GW34 (per-region file):** EN-IT-UL, EN-IT-DL, EN-ET, IN, Astrocyte, EC, Glia (the file's native taxonomy)
- **Harmonization function:** Maps master H1/H2 to per-region H1 taxonomy (collapses EN-IT to UL/DL, combines Astro-1 and Astro-late1 into "Astrocyte"; drops RG/IPC/EN-Mig since they don't exist in CP/MZ)

---

## Analytical pipeline (the 2x2 comparison)

### Pipeline stages

**Stage 1 — Reference annotation (varies by pipeline):**
- *Standard:* scVI embedding + Leiden clustering on paired snRNA-seq; manual confirmation against H2 labels
- *FM-enhanced:* UCE embeddings (pre-computed from CELLxGENE Census) or scGPT zero-shot inference on paired snRNA-seq

**Stage 2 — Spatial cell annotation transfer:**
- scANVI or label transfer from annotated reference to MERFISH cells (same method across all pipelines)

**Stage 3 — Spatial domain identification (varies by pipeline):**
- *Standard:* Banksy spatial clustering
- *GNN-enhanced:* STAGATE (default) or GraphST (fallback)

**Stage 4 — Spatial cell-cell communication:**
- COMMOT with the CellChatDB ligand-receptor database
- Run on ENVI-imputed expression (not raw 300-gene MERFISH) to access 
  full L-R pair coverage
- Comparison run: CellChat (non-spatial) on the same cells for context

### The 2x2 pipeline matrix

| | Standard annotation | FM-enhanced annotation |
|---|---|---|
| **Standard spatial (Banksy)** | Baseline pipeline | A-only |
| **GNN spatial (STAGATE)** | B-only | A+B combined |

For each pipeline, identify V1-vs-V2 differential CCC niches. Compare across pipelines:
- Which niches are robust (present in all four pipelines)?
- Which are method-dependent (present in only specific pipelines)?
- What biological interpretation do the differences support?

### Ground truth evaluation

The `area` column in the master file provides published V1/V2 assignments. For each spatial method (Banksy, STAGATE), evaluate:
- Recovery of the sharp V1/V2 boundary (boundary cell precision/recall)
- Agreement with the paper's `area` labels
- Sharpness of the inferred boundary (transition gradient steepness)

---

## Scope locks

To prevent scope creep, the following are explicitly **out of scope**:

- Migration stream analyses (EN-Mig populations in oSVZ/IZ)
- Full subplate subtype CCC analysis (mentioned in discussion only)
- Other cortical area comparisons (PFC, parietal, etc.) — V1/V2 only
- GW15 or GW22 time points
- Cross-species comparisons (macaque, mouse)
- Custom deep learning model development (use established tools only)
- Foundation model fine-tuning (zero-shot or pre-computed embeddings only)

---

## Week-by-week plan

### Week 1 (current) — Setup and verification
- [x] Download MERFISH data from Zenodo
- [x] Verify file structure and annotation taxonomy
- [x] Identify V1/V2 analytical population
- [x] Apply UL/DL harmonization function to master file
- [x] Visualize V1/V2 boundary at GW20 (sanity check)
- [x] Set up environment with all required tools (scVI, Squidpy, COMMOT)
- [x] Test STAGATE installation; verify GPU access if available
- [x] Lock project brief (this document)

**Deliverable:** Locked brief; clean data loading notebook; one verification figure (V1/V2 boundary visualization at GW20)

### Week 2 — Standard pipeline baseline
- [x] Annotate paired snRNA-seq with standard scVI + Leiden
- [x] Transfer annotations to MERFISH V1/V2 cells at GW20
- [x] Run Banksy spatial domain identification
- [x] Validate against paper's area labels
- [x] Run baseline COMMOT analysis (V1 vs V2)
- [x] Identify baseline V1-vs-V2 differential CCC niches

**Deliverable:** Baseline pipeline complete; first list of V1/V2 differential niches; one summary figure

### Week 3 — Foundation model and GNN pipelines
- [x] Pull UCE embeddings (or run scGPT inference) on paired snRNA-seq
- [x] Compare FM annotations to standard annotations (where do they differ?)
- [x] Transfer FM annotations to MERFISH cells
- [x] Run STAGATE on MERFISH V1/V2 sections
- [x] Evaluate spatial domain agreement with `area` labels for both Banksy and STAGATE

**Deliverable:** FM and GNN pipeline outputs ready for comparison

### Week 4 — Full 2x2 comparison and biological interpretation
- [x] Run COMMOT on all four pipelines
- [x] Identify robust vs. method-dependent V1-vs-V2 CCC niches
- [ ] Biological interpretation: connect findings to known signaling pathways (Wnt, Notch, FGF, EphrinB)
- [ ] Visium validation: confirm key L-R pairs in whole-transcriptome data
- [x] Decision point: is GW34 extension on track?

**Deliverable:** Complete GW20 analysis; first draft of main figures; decision on GW34 extension

### Week 5 — GW34 extension (if on track) and SP angle
- [ ] Apply harmonization to per-region BA17 file
- [ ] Run best-performing pipeline (from week 4 results) on GW34 V1/V2 data
- [ ] Ask: do GW20 V1-vs-V2 niches persist at GW34?
- [ ] Brief SP subtype observation within V1/V2 EN-ET cells at GW20

**Deliverable:** GW34 comparison results; SP discussion paragraph drafted

### Week 6 — Writeup, figures, repo polish
- [ ] Draft 2,500–4,000-word writeup (blog post or short preprint format)
- [ ] Finalize 4–6 publication-quality figures
- [ ] Clean GitHub repo: README (with DAG), environment.yml, download script
- [ ] Productionize: Nextflow, Docker, tests
- [ ] Optional: bioRxiv preprint if results warrant

**Deliverable:** Public repo + writeup; "elevator paragraph" version for applications

---

## Success criteria

The project succeeds if it produces:

1. A clear answer to whether V1 and V2 differ in their spatial CCC niches at GW20 (positive or negative finding both acceptable)
2. A characterized comparison of how foundation model and GNN methods change CCC results
3. A reproducible GitHub repository with environment file, download script, and well-documented notebooks
4. A coherent ~3,000-word writeup that communicates the work clearly to both biological and ML audiences

The project is **not** required to find dramatically novel biology to succeed. A careful, well-executed null result on the methodological question ("FMs and GNNs don't substantively change CCC inference for this question") is a legitimate finding worth publishing.

---

## Kill criteria

If any of the following occur, the corresponding fallback applies:

| Trigger | When detected | Fallback |
|---|---|---|
| MERFISH preprocessing eats more than week 1 | End of week 1 | Use only the per-region files; lose multi-area master analysis |
| STAGATE installation fails irrecoverably | Day 2–3 of week 1 | Use GraphST; if that fails, drop Option B entirely (single pipeline comparison: standard vs FM-annotation) |
| FM embeddings produce annotations identical to standard | End of week 3 | Drop Option A as primary methodological thread; report negative result; deepen biological analysis instead |
| GW20 V1/V2 CCC analysis incomplete by end of week 4 | Mid week 4 | Drop GW34 extension; deliver GW20-only project |
| Compute infrastructure unavailable | Week 1 | Switch to CPU-only methods (Banksy + scVI baseline only); drop FM/GNN pipelines |
| Per-region BA17 file is unusable | Week 5 | Drop GW34 extension; expand SP discussion |

---

## Open questions to resolve early

These are practical questions that should be settled in week 1 or early week 2:

1. **GPU access:** Confirm availability of GPU resources (university cluster, cloud credits, or pay-as-you-go). Verify STAGATE training is feasible on chosen infrastructure.
2. **UCE embeddings for Braun:** Check CELLxGENE Census for pre-computed UCE embeddings on Braun et al. If unavailable, switch to scGPT or accept additional GPU cost.
3. **Visium section identity:** Confirm which tissue section the Visium data corresponds to. If matched to MERFISH section, full cross-modality validation is possible.
4. **Cell subsampling strategy:** Decide how to subsample V1/V2 cells for COMMOT compute. Stratify by area, layer, and cell type to maintain biological structure.

---

## Tools and environment

**Core:** Python 3.11, Scanpy, AnnData, Squidpy, scVI-tools, scenvi

**Foundation models:** scGPT zero-shot (instead of UCE via CELLxGENE Census)

**Spatial methods:** Banksy (standard), STAGATE (GNN primary), GraphST (GNN fallback)

**CCC:** COMMOT (spatial), CellChat (non-spatial context)

**R interop (for .rds files):** zellkonverter or anndata2ri

**Other:** Scipy, scikit-learn for stats/ML; matplotlib, seaborn for plotting; pandas, numpy for data wrangling

Environment pinned in `environment.yml`.

---

## References (key papers)

1. **Qian et al. 2025** *Nature* — primary dataset; V1/V2 boundary finding
2. **Cang et al. 2023** *Nature Methods* — COMMOT (spatial CCC method)
3. **Singhal et al. 2024** *Nature Genetics* — Banksy (standard spatial clustering)
4. **Dong & Zhang 2022** *Nature Communications* — STAGATE (GNN spatial method)
5. **Lopez et al. 2018** *Nature Methods* — scVI (standard annotation)
6. **Cui et al. 2024** *Nature Methods* — scGPT (foundation model)
7. **Rosen et al. 2023** *bioRxiv* — UCE (universal cell embedding)
8. **Jin et al. 2021** *Nature Communications* — CellChat (non-spatial CCC reference)
9. **Braun et al. 2023** *Science* — developmental brain reference dataset
10. **Almet et al. 2021** *Curr Opin Sys Biol* — review of CCC inference (critical perspective)

---

## Decision log

Record significant decisions and pivots as the project evolves. Date each entry.

- *2026-05-29* — Project brief locked; primary biological question is V1/V2 signaling at GW20 with GW34 persistence as longitudinal extension.
- *2026-06* — ENVI imputation coverage fix: the original HVG-only imputation (`envi_FB080`, 2244 genes) reached only 2.2% CellChatDB ligand-receptor pair coverage — unusable for COMMOT. Re-ran with `--lr-database cellchat` (`envi_FB080_lr`, 2931 genes: HVGs ∪ L-R genes ∪ panel overlap), reaching 83.1% pair coverage (1611/1939). This is the production imputation for all downstream COMMOT/niche work; the old HVG-only run is kept only for comparison. 5-fold leave-genes-out CV showed ENVI beats kNN/cell-type-mean baselines only modestly (+0.016 median Spearman) — a "modest improvement" result the project brief explicitly accepts.
- *2026-07-02 to 2026-07-09* — COMMOT run per-region (V1 and V2 separately), not combined; confirmed the A/B tissue-piece labels are not a biological confound (the two pieces sit ~1.7 mm apart, far beyond any CCC distance threshold, so no cross-piece signaling is possible either way). Execution redesigned mid-stream as a pair-parallel array job (not cell subsampling) after the first full-chunk launch OOM-killed 23/24 chunks — `run_commot.py --pair-batch` now caps peak memory by processing L-R pairs a few at a time instead of holding cell×cell transport matrices for an entire chunk simultaneously.
- *2026-07-09 to 2026-07-22* — 2x2 niche-comparison pipeline framing locked in: all four pipeline arms ({scVI, scGPT} annotation × {Banksy, STAGATE} spatial domains) share one fixed COMMOT signaling substrate; ENVI imputation is baseline preprocessing, not a methodological variable under comparison. Primary domain count k=8, with k=6-14 swept for robustness.
- *2026-07-22* — First full 2x2 comparison pass complete, descriptive only (no significance testing yet): 7/8 niches robust across Banksy/STAGATE; STAGATE niches match better across V1/V2 than Banksy's; Q1 (does annotation method change niche composition?) = apparent YES, but confounded by scVI (37 Leiden clusters) vs scGPT (14 clusters) at matched nominal resolution.
- *2026-07-23 to 2026-07-25* — Hardening pass on the 2x2 comparison: added permutation significance testing (1000 permutations) and extended the k-sweep across the full harness. The "niches robust across methods" finding is now statistically significant (p<0.05) at every k from 6-14, not just k=8. Added a dedicated subplate (SP) signaling analysis: confirmed subplate is robustly EN-ET-neuron-dominated and confirmed the V2↑ECM/guidance signaling story, but found the earlier "V1↑WNT" claim does not clearly hold up under direct SP-cell inspection — flagged as needing a recheck before citing further.
- *2026-07-25* — Resolved the Q1 annotation-confound (scVI-vs-scGPT clustering granularity), converging on the same answer via two independent methods: resolution-matching scGPT's Leiden clustering to scVI's cluster count, and building an explicit fine-to-coarse cell-type mapping. Both substantially raise composition agreement (medians rise from ~0.52-0.77 to ~0.65-0.94), confirming most of the original Q1 finding was a clustering-granularity artifact rather than a true scVI-vs-scGPT embedding difference. A real residual disagreement persists on a handful of domains dominated by closely related EN-IT excitatory-neuron subtypes — scGPT's zero-shot embedding genuinely struggles there relative to scVI. Also surfaced but not yet fixed: `annotate_cells.py`'s low-confidence flagging is driven almost entirely by a fixed margin threshold that's miscalibrated across the board (59-84% of clusters flagged low-confidence regardless of fine or coarse labels) — noted as follow-up work.
- *2026-07-26* — Deferred the cell-type-aware STAGATE sensitivity check (originally scoped as a fourth hardening item) as lowest priority; not started.
- *2026-07-26* — Added static-figure and interactive-dashboard visualizations of the 2x2 comparison outputs: cross-method domain agreement across k, cross-region matching and significance by arm, and the Q1 annotation-axis composition-cosine story (including a per-domain "chance level" reference line from the permutation null, added to make clear why moderate cosine values can still register as statistically significant in some regions/domains but not others).

---
