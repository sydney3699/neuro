#!/usr/bin/env python
"""
Build the NNLS-transfer niche-comparison dashboard.

Replaces the hand-built `niche_comparison_dashboard.html`, which hardcoded
single-run Q1 composition cosines (0.5203, 0.6066, ...) that the seed-variance
analysis showed are not reproducible to the precision shown. Nothing generated the
old page, so it could not be refreshed when the numbers changed -- hence this
script.

Reads only committed result CSVs (no cluster access) and emits a self-contained
page: no external scripts, no CDN, light/dark via tokens, every value direct-labeled
(the palette's contrast WARN on the lighter series obliges labels, per the dataviz
skill), and a table view alongside every chart so identity is never color-alone.

Palette is the one the previous dashboard already used, re-validated for the pairs
actually placed adjacent here (blue/green for baseline-vs-NNLS, amber/pink for
scVI-vs-scGPT) in both modes.
"""
import argparse
import csv
import html
import statistics as st
from collections import defaultdict
from pathlib import Path

PALETTE = {
    "baseline": ("#2a78d6", "#3987e5"),   # (light, dark)
    "nnls": ("#1baf7a", "#199e70"),
    "scvi": ("#eda100", "#c98500"),
    "scgpt": ("#e87ba4", "#d55181"),
}


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def esc(s):
    return html.escape(str(s))


def bar(frac, color_var, width=120):
    """A thin horizontal bar; data-end rounded, anchored at the baseline."""
    w = max(0.0, min(1.0, frac)) * width
    return (f'<span class="bar-track" style="width:{width}px">'
            f'<span class="bar-fill" style="width:{w:.1f}px;background:var({color_var})"></span>'
            f'</span>')


def range_bar(lo, hi, vmin, vmax, color_var, width=150):
    """Span bar showing a min-max range on a shared scale -- the seed error bar."""
    span = max(vmax - vmin, 1e-9)
    x0 = (lo - vmin) / span * width
    x1 = (hi - vmin) / span * width
    return (f'<span class="bar-track" style="width:{width}px">'
            f'<span class="bar-span" style="left:{x0:.1f}px;width:{max(x1 - x0, 2):.1f}px;'
            f'background:var({color_var})"></span></span>')


def main():
    p = argparse.ArgumentParser(description="Build the NNLS niche-comparison dashboard")
    p.add_argument("--results", default="results_local", help="dir holding the result subdirs")
    p.add_argument("--out", default="results_local/niche_comparison_nnls_dashboard.html")
    args = p.parse_args()

    R = Path(args.results)
    q1 = read_csv(R / "niche_comparison_nnls" / "nnls_annotation_axis_summary.csv")
    base_var = read_csv(R / "q1_seed_variance" / "q1_seed_summary.csv")
    xreg = read_csv(R / "niche_comparison_nnls" / "nnls_crossregion_match.csv")
    mass = read_csv(R / "niche_comparison_nnls" / "nnls_label_mass.csv")
    ari = read_csv(R / "niche_comparison" / "crossmethod_ari_by_k.csv")
    seedstab = read_csv(R / "seed_stability" / "seed_verdict.csv")
    nnlsstab = read_csv(R / "transfer_fix" / "nnls_seed_stability.csv")

    basemap = {(r["region"], r["method"]): r for r in base_var}

    # ---- labels recovered only by the mixture weights ----
    only = defaultdict(set)
    for r in mass:
        if r["recovered_only_by_weights"] == "True":
            only[r["arm"]].add(r["label"])

    # ---- EN-IT weighted vs hard mass, averaged over seeds ----
    enit = defaultdict(list)
    for r in mass:
        if r["label"].startswith("EN-IT"):
            enit[(r["arm"], r["label"])].append(
                (float(r["weighted_mass"]), float(r["hard_mass"])))
    enit_labels = sorted({l for _, l in enit})

    # ---- cross-region, aggregated per method/arm ----
    xagg = defaultdict(list)
    for r in xreg:
        xagg[(r["method"], r["arm"])].append(int(r["n_mutual_best"]))

    # ---- baseline vs NNLS EN-IT recovery, for the stability section ----
    base_recov = {(r["group"], r["label"]): r for r in seedstab}
    nnls_recov = {(r["arm"], r["label"]): r for r in nnlsstab}

    css = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#898781;--grid:#e1e0d9;--baseline:#c3c2b7;
--border:rgba(11,11,11,0.10);--c-baseline:#2a78d6;--c-nnls:#1baf7a;--c-scvi:#eda100;
--c-scgpt:#e87ba4;--status-good:#0ca30c;--status-critical:#d03b3b;--track:#eceae3;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .viz-root{
color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#fff;
--text-secondary:#c3c2b7;--text-muted:#898781;--grid:#2c2c2a;--baseline:#383835;
--border:rgba(255,255,255,0.10);--c-baseline:#3987e5;--c-nnls:#199e70;--c-scvi:#c98500;
--c-scgpt:#d55181;--status-good:#0ca30c;--status-critical:#e66767;--track:#242423;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#898781;--grid:#2c2c2a;
--baseline:#383835;--border:rgba(255,255,255,0.10);--c-baseline:#3987e5;--c-nnls:#199e70;
--c-scvi:#c98500;--c-scgpt:#d55181;--status-good:#0ca30c;--status-critical:#e66767;--track:#242423;}
body{margin:0;background:var(--page);}
.viz-root{background:var(--page);color:var(--text-primary);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:32px 24px 64px;max-width:1080px;margin:0 auto;}
h1{font-size:24px;margin:0 0 6px;letter-spacing:-0.01em;}
h2{font-size:17px;margin:38px 0 4px;letter-spacing:-0.005em;}
h3{font-size:14px;margin:20px 0 6px;color:var(--text-secondary);font-weight:600;}
.sub{color:var(--text-secondary);margin:0 0 4px;}
.muted{color:var(--text-muted);font-size:12.5px;}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
padding:16px 18px;margin:12px 0;}
.callout{border-left:3px solid var(--c-nnls);padding:10px 14px;margin:14px 0;
background:var(--surface-1);border-radius:0 8px 8px 0;}
.callout.warn{border-left-color:var(--status-critical);}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);
vertical-align:middle;white-space:nowrap;}
th{color:var(--text-secondary);font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:0.04em;}
td.num{font-variant-numeric:tabular-nums;}
.bar-track{display:inline-block;position:relative;height:9px;background:var(--track);
border-radius:5px;vertical-align:middle;overflow:hidden;}
.bar-fill{display:block;height:9px;border-radius:0 4px 4px 0;}
.bar-span{position:absolute;top:0;height:9px;border-radius:4px;}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 2px;font-size:12.5px;
color:var(--text-secondary);}
.key{display:inline-flex;align-items:center;gap:6px;}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block;}
.scroll{overflow-x:auto;}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;
border:1px solid var(--border);color:var(--text-secondary);}
.good{color:var(--status-good);font-weight:600;}
.bad{color:var(--status-critical);font-weight:600;}
code{font-size:12.5px;background:var(--track);padding:1px 5px;border-radius:4px;}
"""

    H = []
    A = H.append
    A('<div class="viz-root">')
    A('<h1>Niche comparison under the NNLS label transfer</h1>')
    A('<p class="sub">V1 vs V2 &middot; scVI/scGPT annotation &times; Banksy/STAGATE domains '
      '&middot; k=8 &middot; 5 seeds per arm</p>')
    A('<p class="muted">Regenerated from the NNLS deconvolution transfer. Supersedes the '
      'previous dashboard, whose Q1 composition cosines were single-run point estimates '
      'later shown to vary by 0.20&ndash;0.36 across seeds alone.</p>')

    # ---------- why ----------
    A('<div class="callout"><strong>Why these numbers moved.</strong> The baseline transfer '
      'assigned each Leiden cluster one label by <code>argmax</code> correlation to a reference '
      'centroid. Cells of the contested EN-IT subtypes never exceeded ~26% of any scGPT cluster, '
      'so that argmax could never land on them. NNLS (non-negative least squares) instead fits '
      'each cluster mean as a <em>non-negative mixture</em> of all 35 reference centroids, so a '
      'type present at 26% surfaces as a 0.26 weight rather than vanishing.</div>')

    # ---------- 1. mixture vs hard labels ----------
    A('<h2>1. Composition must read the mixture, not the hard labels</h2>')
    A('<p class="sub">NNLS still writes one hard label per cell (the mixture\'s argmax) for '
      'downstream compatibility &mdash; but in the scGPT arm that argmax <em>still</em> never '
      'lands on the contested types. Building composition from hard labels would therefore '
      'recover nothing. All composition below is the mean mixture vector per domain.</p>')
    A('<div class="card">')
    A('<h3>Mean composition mass per EN-IT subtype &mdash; weighted vs hard</h3>')
    A('<div class="legend">'
      f'<span class="key"><span class="dot" style="background:var(--c-scvi)"></span>scVI</span>'
      f'<span class="key"><span class="dot" style="background:var(--c-scgpt)"></span>scGPT</span>'
      '<span class="key muted">bars = weighted (mixture) mass</span></div>')
    A('<div class="scroll"><table><thead><tr><th>EN-IT subtype</th>'
      '<th colspan="2">scVI weighted</th><th class="num">scVI hard</th>'
      '<th colspan="2">scGPT weighted</th><th class="num">scGPT hard</th>'
      '<th>recovered only by weights</th></tr></thead><tbody>')
    vmax = max((st.mean(x[0] for x in v) for v in enit.values()), default=1) or 1
    for l in enit_labels:
        A(f'<tr><td>{esc(l)}</td>')
        flags = []
        for arm, var in (("scvi", "--c-scvi"), ("scgpt", "--c-scgpt")):
            v = enit.get((arm, l))
            if not v:
                A('<td></td><td class="num">&mdash;</td><td class="num">&mdash;</td>')
                continue
            w = st.mean(x[0] for x in v)
            h = st.mean(x[1] for x in v)
            A(f'<td>{bar(w / vmax, var, 90)}</td><td class="num">{w:.4f}</td>'
              f'<td class="num">{h:.4f}</td>')
            if w > 0.01 and h == 0.0:
                flags.append(arm)
        tag = (f'<span class="pill good">{" + ".join(flags)}</span>' if flags else
               '<span class="muted">&mdash;</span>')
        A(f'<td>{tag}</td></tr>')
    A('</tbody></table></div>')
    A(f'<p class="muted">Across all 35 reference types: '
      f'<strong>{len(only["scvi"])}</strong> carry &gt;1% weighted mass but zero hard-label mass '
      f'in the scVI arm, <strong>{len(only["scgpt"])}</strong> in the scGPT arm &mdash; including '
      f'EN-IT-L4-V1, EN-IT-DL-1 and EN-IT-UL-1, the exact types the baseline lost.</p>')
    A('</div>')

    # ---------- 2. Q1 ----------
    A('<h2>2. Q1 &mdash; does the annotation method change niche composition?</h2>')
    A('<div class="callout warn"><strong>Do not compare the raw cosines across '
      'representations.</strong> NNLS composition vectors are dense over all 35 types; the '
      'baseline\'s were sparse over each arm\'s reached labels only (scVI 16&ndash;17, scGPT '
      '10&ndash;13) with partly disjoint support. Cosine between two dense vectors is '
      'systematically higher regardless of whether the arms agree better. The '
      '<strong>matched&minus;mismatched gap</strong> below cancels that and is the figure to cite.'
      '</div>')
    A('<div class="card">')
    A('<h3>Gap = median cos(scVI[d], scGPT[d]) &minus; mean cos(scVI[d], scGPT[d&prime;]), d&ne;d&prime;</h3>')
    A('<div class="legend">'
      '<span class="key"><span class="dot" style="background:var(--c-baseline)"></span>'
      'baseline (hard labels, 1 run)</span>'
      '<span class="key"><span class="dot" style="background:var(--c-nnls)"></span>'
      'NNLS (weighted, 25 seed pairs)</span></div>')
    gaps = [float(r["nnls_gap_median"]) for r in q1] + [float(r["baseline_gap"]) for r in q1]
    gmin, gmax = 0.0, max(gaps) * 1.1
    A('<div class="scroll"><table><thead><tr><th>region &times; method</th>'
      '<th colspan="2">baseline gap</th><th colspan="2">NNLS gap (median)</th>'
      '<th class="num">NNLS gap range</th><th>verdict</th></tr></thead><tbody>')
    for r in q1:
        bg = float(r["baseline_gap"]); ng = float(r["nnls_gap_median"])
        lo, hi = float(r["nnls_gap_min"]), float(r["nnls_gap_max"])
        same = "higher" if ng > bg else "lower"
        A(f'<tr><td>{esc(r["region"])} &times; {esc(r["method"])}</td>'
          f'<td>{bar((bg - gmin) / (gmax - gmin), "--c-baseline", 100)}</td>'
          f'<td class="num">{bg:.3f}</td>'
          f'<td>{bar((ng - gmin) / (gmax - gmin), "--c-nnls", 100)}</td>'
          f'<td class="num">{ng:.3f}</td>'
          f'<td class="num">{lo:.3f}&ndash;{hi:.3f}</td>'
          f'<td class="muted">NNLS {same}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="muted">The effect itself is essentially unchanged &mdash; NNLS is higher in V1, '
      'lower in V2, with no consistent direction. What changes is whether it can be '
      '<em>measured</em>.</p>')
    A('</div>')

    # ---------- 3. the real improvement: reproducibility ----------
    A('<div class="card">')
    A('<h3>The real gain: the Q1 statistic becomes reproducible</h3>')
    A('<p class="sub">Seed-to-seed spread of the composition agreement, baseline vs NNLS. The '
      'baseline figure varies only the scGPT seed (10 seeds, one arm); NNLS varies '
      '<em>both</em> arms (25 seed pairs) and is still 3&ndash;5&times; tighter.</p>')
    allr = ([float(basemap[(r["region"], r["method"])]["seed_range"]) for r in q1]
            + [float(r["nnls_matched_cosine_range"]) for r in q1])
    rmax = max(allr) * 1.1
    A('<div class="scroll"><table><thead><tr><th>region &times; method</th>'
      '<th colspan="2">baseline spread (1 arm reseeded)</th>'
      '<th colspan="2">NNLS spread (both arms)</th><th class="num">tighter by</th>'
      '</tr></thead><tbody>')
    for r in q1:
        b = float(basemap[(r["region"], r["method"])]["seed_range"])
        n = float(r["nnls_matched_cosine_range"])
        A(f'<tr><td>{esc(r["region"])} &times; {esc(r["method"])}</td>'
          f'<td>{bar(b / rmax, "--c-baseline", 110)}</td><td class="num">{b:.3f}</td>'
          f'<td>{bar(n / rmax, "--c-nnls", 110)}</td><td class="num">{n:.3f}</td>'
          f'<td class="num good">{b / n:.1f}&times;</td></tr>')
    A('</tbody></table></div>')
    A('<p class="muted">Under the baseline, seed noise (0.203&ndash;0.361) was as large as the '
      'effect being claimed, so Q1 could not be stated as a number. Under NNLS the noise '
      '(0.043&ndash;0.108) is well below the gap (0.35&ndash;0.48), so it can.</p>')
    A('</div>')

    # ---------- 4. label recovery stability ----------
    A('<h2>3. Label recovery &mdash; baseline vs NNLS, across seeds</h2>')
    A('<div class="card">')
    A('<p class="sub">Fraction of seeds in which the transfer reaches each contested EN-IT '
      'subtype. Baseline: 10 Leiden seeds. NNLS: 5 seeds.</p>')
    A('<div class="scroll"><table><thead><tr><th>subtype</th>'
      '<th class="num">baseline scVI</th><th class="num">baseline scGPT</th>'
      '<th class="num">NNLS scVI</th><th class="num">NNLS scGPT</th>'
      '<th class="num">NNLS scGPT weight</th></tr></thead><tbody>')
    for l in enit_labels:
        bs = base_recov.get(("annot_seed_scvi_leiden", l))
        bg = base_recov.get(("annot_seed_scgpt_leiden", l))
        ns = nnls_recov.get(("scvi", l))
        ng = nnls_recov.get(("scgpt", l))

        def cell(d, key="reach_frac"):
            if not d:
                return '<td class="num muted">&mdash;</td>'
            v = float(d[key])
            cls = "good" if v == 1.0 else ("bad" if v == 0.0 else "")
            return f'<td class="num {cls}">{v:.1f}</td>'
        w = (f'{float(ng["mean_max_weight"]):.3f}' if ng else "&mdash;")
        A(f'<tr><td>{esc(l)}</td>{cell(bs)}{cell(bg)}{cell(ns)}{cell(ng)}'
          f'<td class="num muted">{w}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="muted">NNLS reaches every contested subtype in both arms at every seed, and '
      'recovers EN-IT-UL-2/-3/-4 in the scVI arm &mdash; labels no arm reached under the '
      'baseline at any Leiden resolution. EN-IT-UL-5 remains unreached everywhere. '
      'Read the weight column alongside: in the scGPT arm these types are recovered as '
      '<em>minor</em> components (0.13&ndash;0.19), never a cluster\'s dominant type.</p>')
    A('</div>')

    # ---------- 5. cross-region ----------
    A('<h2>4. Cross-region V1-vs-V2 niche matching</h2>')
    A('<div class="card">')
    A('<p class="sub">Mutual-best-matched niches out of 8, mean over 5 seeds.</p>')
    A('<div class="legend">'
      '<span class="key"><span class="dot" style="background:var(--c-scvi)"></span>scVI</span>'
      '<span class="key"><span class="dot" style="background:var(--c-scgpt)"></span>scGPT</span>'
      '</div>')
    A('<div class="scroll"><table><thead><tr><th>method</th><th>arm</th>'
      '<th colspan="2">mutual-best of 8</th><th class="num">range</th></tr></thead><tbody>')
    for (m, arm), vals in sorted(xagg.items()):
        var = "--c-scvi" if arm == "scvi" else "--c-scgpt"
        mu = st.mean(vals)
        A(f'<tr><td>{esc(m)}</td><td>{esc(arm)}</td>'
          f'<td>{bar(mu / 8, var, 110)}</td><td class="num">{mu:.1f}</td>'
          f'<td class="num">{min(vals)}&ndash;{max(vals)}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="muted">The previous dashboard\'s 0.6 cosine threshold is <strong>not '
      'discriminating here</strong>: under dense NNLS composition every mutual-best pair scores '
      '&gt;0.9, so the threshold passes everything. Only the match <em>count</em> is '
      'informative at this representation.</p>')
    A('</div>')

    # ---------- 6. unchanged ----------
    A('<h2>5. Cross-method domain agreement &mdash; unchanged</h2>')
    A('<div class="card">')
    A('<p class="sub">Banksy vs STAGATE agreement is computed on domain cell-membership and is '
      '<strong>annotation-independent</strong>, so the transfer rewrite does not affect it. '
      'Carried forward unchanged.</p>')
    A('<div class="scroll"><table><thead><tr><th>region</th><th>k</th>'
      '<th colspan="2">ARI (Banksy vs STAGATE)</th><th class="num">p</th>'
      '</tr></thead><tbody>')
    for r in ari:
        if r["k"] not in ("6", "8", "10", "12", "14"):
            continue
        v = float(r["ari"])
        A(f'<tr><td>{esc(r["region"])}</td><td class="num">{esc(r["k"])}</td>'
          f'<td>{bar(v, "--c-baseline", 110)}</td><td class="num">{v:.3f}</td>'
          f'<td class="num">{esc(r["p_value_ari"])}</td></tr>')
    A('</tbody></table></div>')
    A('</div>')

    A('<p class="muted" style="margin-top:28px">Generated by '
      '<code>scripts/viz/build_nnls_dashboard.py</code> from the CSVs in '
      '<code>results_local/</code>. Regenerate after any transfer or seed change.</p>')
    A('</div>')

    doc = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
           "<title>Niche Comparison under NNLS Transfer — V1 vs V2</title>\n"
           f"<style>{css}</style>\n</head>\n<body>\n" + "\n".join(H) + "\n</body>\n</html>\n")
    out = Path(args.out)
    out.write_text(doc)
    print(f"wrote {out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
