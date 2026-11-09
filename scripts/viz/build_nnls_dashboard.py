#!/usr/bin/env python
"""
Build the NNLS-transfer niche-comparison dashboard.

Replaces the hand-built `niche_comparison_dashboard.html`, which hardcoded
single-run Q1 composition cosines (0.5203, 0.6066, ...) that the seed-variance
analysis showed are not reproducible to the precision shown. Nothing generated the
old page, so it could not be refreshed when the numbers changed -- hence this script.

Reads only committed result CSVs (no cluster access). Two output modes:

  --mode standalone  (default) a complete .html document, for results_local/
  --mode artifact    page content only -- no <!DOCTYPE>/<html>/<head>/<body>, since
                     the Artifact host supplies that skeleton and only scans the
                     first 8KB for <title>

Both modes emit the same body, so the published page and the repo copy cannot drift.
Fonts come from Google Fonts (the one font host the Artifact CSP admits); everything
else is inline, so the page has no other external dependency.
"""
import argparse
import csv
import html
import statistics as st
from collections import defaultdict
from pathlib import Path

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=IBM+Plex+Mono:wght@400;500;600&"
         "family=IBM+Plex+Sans:wght@400;500;600&"
         "family=IBM+Plex+Serif:wght@500;600&display=swap")


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def esc(s):
    return html.escape(str(s))


def bar(frac, var, width=100):
    w = max(0.0, min(1.0, frac)) * width
    return (f'<span class="track" style="width:{width}px">'
            f'<span class="fill" style="width:{w:.1f}px;background:var({var})"></span></span>')


CSS = """
:root{
  --ground:#faf8f4; --surface:#ffffff; --sunken:#f1eee7;
  --ink:#171614; --ink-2:#56534c; --ink-3:#8b877e;
  --rule:#e3e0d7; --rule-strong:#cdc8ba;
  --baseline:#2a78d6; --nnls:#1baf7a; --scvi:#b37800; --scgpt:#c0577f;
  --good:#0a7d35; --critical:#c0392b;
  --shadow:0 1px 2px rgba(23,22,20,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#131210; --surface:#1c1b18; --sunken:#242219;
    --ink:#f5f3ee; --ink-2:#c2bdb1; --ink-3:#8b877e;
    --rule:#2e2b25; --rule-strong:#413d34;
    --baseline:#5e9ff0; --nnls:#3cc492; --scvi:#d9a440; --scgpt:#e18cad;
    --good:#4cc46a; --critical:#f0776a;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#131210; --surface:#1c1b18; --sunken:#242219;
  --ink:#f5f3ee; --ink-2:#c2bdb1; --ink-3:#8b877e;
  --rule:#2e2b25; --rule-strong:#413d34;
  --baseline:#5e9ff0; --nnls:#3cc492; --scvi:#d9a440; --scgpt:#e18cad;
  --good:#4cc46a; --critical:#f0776a;
  --shadow:none;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:15px;line-height:1.6;}
.wrap{max-width:980px;margin:0 auto;padding:56px 28px 80px;}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:34px;
  line-height:1.15;margin:0 0 10px;letter-spacing:-0.015em;text-wrap:balance;}
h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:21px;
  margin:0 0 6px;letter-spacing:-0.01em;text-wrap:balance;}
h3{font-size:14px;font-weight:600;margin:0 0 8px;color:var(--ink-2);}
p{margin:0 0 10px;max-width:68ch;}
.lede{font-size:16.5px;color:var(--ink-2);max-width:66ch;margin:0 0 4px;}
.meta{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;
  color:var(--ink-3);letter-spacing:0.02em;margin:0 0 40px;}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;
  font-weight:500;letter-spacing:0.13em;text-transform:uppercase;color:var(--ink-3);
  display:block;margin:0 0 7px;}
section{margin:0 0 46px;}
.sec-head{border-top:1px solid var(--rule-strong);padding-top:14px;margin-bottom:16px;}
.note{color:var(--ink-2);font-size:14px;max-width:68ch;margin:10px 0 0;}
.verdict{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:3px;
  overflow:hidden;margin:0 0 44px;}
.vcell{background:var(--surface);padding:18px 20px;}
.vlabel{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--ink-3);display:block;
  margin-bottom:8px;}
.vnum{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:25px;
  font-weight:500;letter-spacing:-0.02em;display:block;line-height:1.1;
  font-variant-numeric:tabular-nums;}
.vwas{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;
  color:var(--ink-3);display:block;margin-top:6px;}
.flag{border-left:2px solid var(--critical);background:var(--sunken);
  padding:13px 16px;margin:16px 0;border-radius:0 3px 3px 0;}
.flag.ok{border-left-color:var(--nnls);}
.flag p{margin:0;font-size:14px;max-width:72ch;}
.scroll{overflow-x:auto;margin:14px 0 0;}
table{border-collapse:collapse;width:100%;font-size:13.5px;}
th,td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--rule);
  white-space:nowrap;vertical-align:middle;}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--ink-3);
  border-bottom:1px solid var(--rule-strong);padding-bottom:8px;}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums;}
tr:last-child td{border-bottom:none;}
.track{display:inline-block;position:relative;height:7px;background:var(--sunken);
  border-radius:4px;vertical-align:middle;overflow:hidden;}
.fill{display:block;height:7px;border-radius:0 3px 3px 0;}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:11.5px;color:var(--ink-2);margin:12px 0 0;}
.key{display:inline-flex;align-items:center;gap:7px;}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block;flex:none;}
.tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
  padding:1px 7px;border:1px solid var(--rule-strong);border-radius:2px;
  color:var(--ink-2);white-space:nowrap;}
.tag.hit{color:var(--good);border-color:var(--good);}
.good{color:var(--good);} .bad{color:var(--critical);}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;
  background:var(--sunken);padding:1px 5px;border-radius:2px;}
footer{border-top:1px solid var(--rule-strong);padding-top:14px;margin-top:8px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--ink-3);}
@media (max-width:720px){
  .verdict{grid-template-columns:1fr;} h1{font-size:27px;} .wrap{padding:36px 18px 56px;}
}
"""


def main():
    p = argparse.ArgumentParser(description="Build the NNLS niche-comparison dashboard")
    p.add_argument("--results", default="results_local")
    p.add_argument("--out", default="results_local/niche_comparison_nnls_dashboard.html")
    p.add_argument("--mode", default="standalone", choices=["standalone", "artifact"])
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
    only = defaultdict(set)
    for r in mass:
        if r["recovered_only_by_weights"] == "True":
            only[r["arm"]].add(r["label"])
    enit = defaultdict(list)
    for r in mass:
        if r["label"].startswith("EN-IT"):
            enit[(r["arm"], r["label"])].append(
                (float(r["weighted_mass"]), float(r["hard_mass"])))
    enit_labels = sorted({l for _, l in enit})
    xagg = defaultdict(list)
    for r in xreg:
        xagg[(r["method"], r["arm"])].append(int(r["n_mutual_best"]))
    base_recov = {(r["group"], r["label"]): r for r in seedstab}
    nnls_recov = {(r["arm"], r["label"]): r for r in nnlsstab}

    gmin_eff = min(float(r["nnls_gap_median"]) for r in q1)
    gmax_eff = max(float(r["nnls_gap_median"]) for r in q1)
    nmin = min(float(r["nnls_matched_cosine_range"]) for r in q1)
    nmax = max(float(r["nnls_matched_cosine_range"]) for r in q1)
    bmin = min(float(r["seed_range"]) for r in base_var)
    bmax = max(float(r["seed_range"]) for r in base_var)

    H = []
    A = H.append
    A('<div class="wrap">')
    A('<h1>Niche comparison, rebuilt on a mixture-based label transfer</h1>')
    A('<p class="lede">V1 vs V2 visual cortex at GW20 &mdash; scVI/scGPT annotation '
      '&times; Banksy/STAGATE domains, k=8 domains, 5 seeds per arm.</p>')
    A('<p class="meta">Supersedes the previous dashboard, whose Q1 figures were '
      'single-run point estimates.</p>')

    # ---- verdict band: summary before detail ----
    A('<div class="verdict">')
    A(f'<div class="vcell"><span class="vlabel">Q1 effect (gap)</span>'
      f'<span class="vnum">{gmin_eff:.2f}&ndash;{gmax_eff:.2f}</span>'
      f'<span class="vwas">essentially unchanged</span></div>')
    A(f'<div class="vcell"><span class="vlabel">Seed noise</span>'
      f'<span class="vnum" style="color:var(--nnls)">{nmin:.2f}&ndash;{nmax:.2f}</span>'
      f'<span class="vwas">was {bmin:.2f}&ndash;{bmax:.2f}</span></div>')
    A('<div class="vcell"><span class="vlabel">Contested subtypes reached</span>'
      '<span class="vnum" style="color:var(--nnls)">3 / 3</span>'
      '<span class="vwas">scGPT was 0/10 seeds</span></div>')
    A('</div>')

    A('<div class="flag ok"><p><strong>The headline.</strong> The Q1 effect itself barely '
      'moved. What moved is whether it can be measured: seed noise that used to be as large '
      'as the effect is now 3&ndash;5&times; smaller than it. Q1 is statable as a number for '
      'the first time.</p></div>')

    # ---- method ----
    A('<section><div class="sec-head"><span class="eyebrow">Method</span>'
      '<h2>Why the numbers moved</h2></div>')
    A('<p>The baseline transfer gave each Leiden cluster one label by <code>argmax</code> '
      'correlation against a reference centroid. Cells of the contested EN-IT subtypes never '
      'exceeded ~26% of any scGPT cluster, so that argmax could never land on them. '
      'Non-negative least squares instead fits each cluster mean as a non-negative mixture of '
      'all 35 reference centroids, so a type present at 26% surfaces as a 0.26 weight rather '
      'than vanishing.</p>')
    A('</section>')

    # ---- audit ----
    A('<section><div class="sec-head"><span class="eyebrow">Audit</span>'
      '<h2>Composition has to read the mixture, not the labels</h2></div>')
    A('<p>NNLS still writes one hard label per cell &mdash; the mixture\'s argmax &mdash; for '
      'downstream compatibility. In the scGPT arm that argmax <em>still</em> never lands on the '
      'contested types, so composition built from hard labels would recover nothing. Every '
      'composition below is the mean mixture vector per domain.</p>')
    A('<div class="legend">'
      '<span class="key"><span class="dot" style="background:var(--scvi)"></span>scVI</span>'
      '<span class="key"><span class="dot" style="background:var(--scgpt)"></span>scGPT</span>'
      '<span class="key">bars show weighted (mixture) mass</span></div>')
    A('<div class="scroll"><table><thead><tr><th>EN-IT subtype</th>'
      '<th colspan="2">scVI weighted</th><th>scVI hard</th>'
      '<th colspan="2">scGPT weighted</th><th>scGPT hard</th>'
      '<th>weights only</th></tr></thead><tbody>')
    vmax = max((st.mean(x[0] for x in v) for v in enit.values()), default=1) or 1
    for l in enit_labels:
        A(f'<tr><td>{esc(l)}</td>')
        flags = []
        for arm, var in (("scvi", "--scvi"), ("scgpt", "--scgpt")):
            v = enit.get((arm, l))
            if not v:
                A('<td></td><td class="n">&mdash;</td><td class="n">&mdash;</td>')
                continue
            w = st.mean(x[0] for x in v)
            h = st.mean(x[1] for x in v)
            A(f'<td>{bar(w / vmax, var, 72)}</td><td class="n">{w:.4f}</td>'
              f'<td class="n">{h:.4f}</td>')
            if w > 0.01 and h == 0.0:
                flags.append(arm)
        tag = (f'<span class="tag hit">{" + ".join(flags)}</span>' if flags
               else '<span class="n" style="color:var(--ink-3)">&mdash;</span>')
        A(f'<td>{tag}</td></tr>')
    A('</tbody></table></div>')
    A(f'<p class="note">Across all 35 reference types, <strong>{len(only["scgpt"])}</strong> '
      f'carry more than 1% weighted mass but zero hard-label mass in the scGPT arm '
      f'(<strong>{len(only["scvi"])}</strong> in scVI) &mdash; including EN-IT-L4-V1, '
      f'EN-IT-DL-1 and EN-IT-UL-1, exactly the types the baseline lost.</p>')
    A('</section>')

    # ---- Q1 ----
    A('<section><div class="sec-head"><span class="eyebrow">Q1 &middot; annotation axis</span>'
      '<h2>Does the annotation method change niche composition?</h2></div>')
    A('<div class="flag"><p><strong>Raw cosines do not compare across the two transfers.</strong> '
      'NNLS composition vectors are dense over all 35 types; the baseline\'s were sparse over '
      'each arm\'s reached labels only (scVI 16&ndash;17, scGPT 10&ndash;13) with partly '
      'disjoint support. Cosine between dense vectors runs higher regardless of whether the '
      'arms agree. The matched&minus;mismatched <em>gap</em> cancels that and is the figure to '
      'cite.</p></div>')
    A('<p class="note">gap = median cos(scVI[d], scGPT[d]) &minus; mean cos(scVI[d], '
      'scGPT[d&prime;]) over d &ne; d&prime;</p>')
    A('<div class="legend">'
      '<span class="key"><span class="dot" style="background:var(--baseline)"></span>'
      'baseline &middot; hard labels &middot; 1 run</span>'
      '<span class="key"><span class="dot" style="background:var(--nnls)"></span>'
      'NNLS &middot; weighted &middot; 25 seed pairs</span></div>')
    gall = ([float(r["nnls_gap_median"]) for r in q1]
            + [float(r["baseline_gap"]) for r in q1])
    gtop = max(gall) * 1.12
    A('<div class="scroll"><table><thead><tr><th>region &times; method</th>'
      '<th colspan="2">baseline gap</th><th colspan="2">NNLS gap</th>'
      '<th>NNLS range</th></tr></thead><tbody>')
    for r in q1:
        bg, ng = float(r["baseline_gap"]), float(r["nnls_gap_median"])
        lo, hi = float(r["nnls_gap_min"]), float(r["nnls_gap_max"])
        A(f'<tr><td>{esc(r["region"])} &times; {esc(r["method"])}</td>'
          f'<td>{bar(bg / gtop, "--baseline", 86)}</td><td class="n">{bg:.3f}</td>'
          f'<td>{bar(ng / gtop, "--nnls", 86)}</td><td class="n">{ng:.3f}</td>'
          f'<td class="n" style="color:var(--ink-3)">{lo:.3f}&ndash;{hi:.3f}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="note">NNLS is higher in V1, lower in V2 &mdash; no consistent direction. '
      'The effect is not what the rewrite changed.</p>')
    A('</section>')

    # ---- reproducibility ----
    A('<section><div class="sec-head"><span class="eyebrow">Reproducibility</span>'
      '<h2>The spread that made Q1 unstatable</h2></div>')
    A('<p>Seed-to-seed spread of composition agreement. The baseline figure varies only the '
      'scGPT seed &mdash; 10 seeds, one arm. NNLS varies <em>both</em> arms across 25 seed '
      'pairs and is still several times tighter.</p>')
    rall = ([float(basemap[(r["region"], r["method"])]["seed_range"]) for r in q1]
            + [float(r["nnls_matched_cosine_range"]) for r in q1])
    rtop = max(rall) * 1.12
    A('<div class="scroll"><table><thead><tr><th>region &times; method</th>'
      '<th colspan="2">baseline &middot; 1 arm reseeded</th>'
      '<th colspan="2">NNLS &middot; both arms</th><th>tighter by</th>'
      '</tr></thead><tbody>')
    for r in q1:
        b = float(basemap[(r["region"], r["method"])]["seed_range"])
        n = float(r["nnls_matched_cosine_range"])
        A(f'<tr><td>{esc(r["region"])} &times; {esc(r["method"])}</td>'
          f'<td>{bar(b / rtop, "--baseline", 92)}</td><td class="n">{b:.3f}</td>'
          f'<td>{bar(n / rtop, "--nnls", 92)}</td><td class="n">{n:.3f}</td>'
          f'<td class="n good">{b / n:.1f}&times;</td></tr>')
    A('</tbody></table></div>')
    A(f'<p class="note">Under the baseline, seed noise ({bmin:.3f}&ndash;{bmax:.3f}) was as '
      f'large as the effect being claimed, so Q1 could not be stated as a number. Under NNLS '
      f'the noise ({nmin:.3f}&ndash;{nmax:.3f}) sits well below the gap '
      f'({gmin_eff:.2f}&ndash;{gmax_eff:.2f}).</p>')
    A('</section>')

    # ---- recovery ----
    A('<section><div class="sec-head"><span class="eyebrow">Label recovery</span>'
      '<h2>Which subtypes each transfer actually reaches</h2></div>')
    A('<p>Fraction of seeds in which the transfer reaches each contested EN-IT subtype. '
      'Baseline: 10 Leiden seeds. NNLS: 5 seeds.</p>')
    A('<div class="scroll"><table><thead><tr><th>subtype</th>'
      '<th>baseline scVI</th><th>baseline scGPT</th>'
      '<th>NNLS scVI</th><th>NNLS scGPT</th>'
      '<th>NNLS scGPT weight</th></tr></thead><tbody>')
    for l in enit_labels:
        cells = [base_recov.get(("annot_seed_scvi_leiden", l)),
                 base_recov.get(("annot_seed_scgpt_leiden", l)),
                 nnls_recov.get(("scvi", l)), nnls_recov.get(("scgpt", l))]
        A(f'<tr><td>{esc(l)}</td>')
        for d in cells:
            if not d:
                A('<td class="n" style="color:var(--ink-3)">&mdash;</td>')
                continue
            v = float(d["reach_frac"])
            cls = "good" if v == 1.0 else ("bad" if v == 0.0 else "")
            A(f'<td class="n {cls}">{v:.1f}</td>')
        ng = cells[3]
        w = f'{float(ng["mean_max_weight"]):.3f}' if ng else "&mdash;"
        A(f'<td class="n" style="color:var(--ink-3)">{w}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="note">NNLS reaches every contested subtype in both arms at every seed, and '
      'recovers EN-IT-UL-2, -3 and -4 in the scVI arm &mdash; labels no arm reached under the '
      'baseline at any Leiden resolution. EN-IT-UL-5 stays unreached everywhere. Read the '
      'weight column alongside: in the scGPT arm these types come back as <em>minor</em> '
      'components (0.13&ndash;0.19), never a cluster\'s dominant type, so this is not evidence '
      'that scGPT resolves them as well as scVI does.</p>')
    A('</section>')

    # ---- cross-region ----
    A('<section><div class="sec-head"><span class="eyebrow">Cross-region</span>'
      '<h2>V1-vs-V2 niche matching</h2></div>')
    A('<p>Mutual-best-matched niches out of 8, averaged over 5 seeds.</p>')
    A('<div class="scroll"><table><thead><tr><th>method</th><th>arm</th>'
      '<th colspan="2">mutual-best of 8</th><th>range</th></tr></thead><tbody>')
    for (m, arm), vals in sorted(xagg.items()):
        var = "--scvi" if arm == "scvi" else "--scgpt"
        mu = st.mean(vals)
        A(f'<tr><td>{esc(m)}</td><td>{esc(arm)}</td>'
          f'<td>{bar(mu / 8, var, 92)}</td><td class="n">{mu:.1f}</td>'
          f'<td class="n" style="color:var(--ink-3)">{min(vals)}&ndash;{max(vals)}</td></tr>')
    A('</tbody></table></div>')
    A('<p class="note">The old 0.6 cosine match threshold is <strong>not discriminating '
      'here</strong>: under dense composition every mutual-best pair scores above 0.9, so the '
      'threshold passes everything. Only the match count carries information at this '
      'representation.</p>')
    A('</section>')

    # ---- unchanged ----
    A('<section><div class="sec-head"><span class="eyebrow">Unchanged</span>'
      '<h2>Cross-method domain agreement</h2></div>')
    A('<p>Banksy vs STAGATE agreement is computed on domain cell-membership and is '
      'annotation-independent, so the transfer rewrite cannot affect it. Carried forward as is.</p>')
    A('<div class="scroll"><table><thead><tr><th>region</th><th>k</th>'
      '<th colspan="2">ARI, Banksy vs STAGATE</th><th>p</th></tr></thead><tbody>')
    for r in ari:
        if r["k"] not in ("6", "8", "10", "12", "14"):
            continue
        v = float(r["ari"])
        A(f'<tr><td>{esc(r["region"])}</td><td class="n">{esc(r["k"])}</td>'
          f'<td>{bar(v, "--baseline", 92)}</td><td class="n">{v:.3f}</td>'
          f'<td class="n" style="color:var(--ink-3)">{esc(r["p_value_ari"])}</td></tr>')
    A('</tbody></table></div>')
    A('</section>')

    A('<footer>Generated by scripts/viz/build_nnls_dashboard.py from the CSVs in '
      'results_local/. Rerun after any transfer or seed change.</footer>')
    A('</div>')

    title = "NNLS Niche Comparison"
    head = (f'<title>{title}</title>\n'
            f'<link rel="stylesheet" href="{FONTS}">\n'
            f'<style>{CSS}</style>\n')
    body = "\n".join(H)

    if args.mode == "artifact":
        doc = head + body + "\n"
    else:
        doc = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
               + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out} ({len(doc)} bytes, mode={args.mode})")


if __name__ == "__main__":
    main()
